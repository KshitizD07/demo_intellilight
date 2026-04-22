"""
IntelliLight 4-Phase Traffic Environment
=======================================

Realistic 4-phase cyclic traffic signal control:

0: EW Through + Right
1: EW Protected Left
2: NS Through + Right
3: NS Protected Left

Improvements:
- Normalized observations
- Queue delta feature
- Cumulative throughput tracking
- Stable reward integration
- Fixed all-red phase handling
- Robust emergency vehicle detection

FIXES APPLIED:
- Removed incorrect all-red phase calls
- More robust emergency direction detection
- Average reward per cycle (not sum)
- Added lane name verification
"""

import os
import sys
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Optional
import traci
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.sumo_env import SUMOSimulation
from simulation.route_generator import RouteGenerator
from rl.reward_function import EnhancedRewardCalculator, RewardWeights


# Configuration
MIN_GREEN = 10
ALL_RED = 4
EPISODE_LENGTH = 1800
TRAFFIC_LIGHT_ID = "J1"

GREEN_DURATIONS = [10, 15, 20, 25, 30, 35, 40, 45]

MAX_QUEUE = 50.0
MAX_WAIT = 120.0


class TrafficEnv4Phase(gym.Env):
    """
    4-Phase Cyclic Traffic Signal Environment.
    
    Action Space:
        MultiDiscrete([8, 8, 8, 8]) - duration index for each of 4 phases
    
    Observation Space:
        Box(14,) - normalized [queues(4), delta(4), waits(4), emergency(1), phase(1)]
    
    One step = complete cycle through all 4 phases
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        use_gui: bool = False,
        episode_length: int = EPISODE_LENGTH,
        curriculum_stage: int = 0,
        seed: Optional[int] = None
    ):
        """
        Initialize 4-phase traffic environment.
        
        Args:
            use_gui: Show SUMO GUI
            episode_length: Episode length in seconds
            curriculum_stage: Curriculum learning stage (0-2)
            seed: Random seed
        """
        super().__init__()
        
        self.use_gui = use_gui
        self.episode_length = episode_length
        self.curriculum_stage = curriculum_stage
        
        # 4-phase configuration
        self.n_phases = 4
        
        self.phase_names = [
            "EW Through",
            "EW Left",
            "NS Through",
            "NS Left"
        ]
        
        # SUMO phase IDs (must match your .net.xml file)
        self.sumo_phases = {
            0: 0,  # EW through
            1: 1,  # EW left
            2: 2,  # NS through
            3: 3   # NS left
        }
        
        # Action space: duration index for each phase
        self.green_durations = GREEN_DURATIONS
        self.n_durations = len(self.green_durations)
        
        self.action_space = spaces.MultiDiscrete([self.n_durations] * self.n_phases)
        
        # Observation space
        # queues(4) + queue_delta(4) + wait_times(4) + emergency(1) + phase(1) = 14
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32
        )
        
        # Initialize SUMO
        self.sumo = SUMOSimulation(use_gui=use_gui)
        self.route_gen = RouteGenerator()
        
        # Initialize reward calculator
        self.reward_calc = EnhancedRewardCalculator(
            weights=RewardWeights(),
            max_acceptable_wait=90,
            emergency_max_wait=30
        )
        
        self.reward_calc.set_curriculum_stage(curriculum_stage)
        
        # State tracking
        self.current_phase = 0
        self.phase_timer = 0
        self.simulation_step = 0
        self.current_step = 0
        self.cycle_count = 0
        
        self.prev_queues = np.zeros(4)
        
        self.emergency_active = False
        self.emergency_direction = -1
        
        self.cumulative_arrived = 0
        
        self.episode_reward = 0.0
        
        # Lane verification flag
        self._lanes_verified = False
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Reset SUMO
        self.sumo.reset()
        
        # Get curriculum scenario
        scenario = self._get_curriculum_scenario()
        
        # Generate route file
        self.current_route_file = self.route_gen.generate_unique_filename()
        scenario_info = self.route_gen.generate_route_file(
            self.current_route_file,
            scenario=scenario,
            curriculum_stage=self.curriculum_stage
        )
        route_file = self.current_route_file
        
        # Start SUMO
        self.sumo.start(route_file)
        
        # Verify lane names on first reset
        if not self._lanes_verified:
            self._verify_lane_names()
            self._lanes_verified = True
        
        # Reset state
        self.current_phase = 0
        self.phase_timer = 0
        self.simulation_step = 0
        self.current_step = 0
        self.cycle_count = 0
        
        self.prev_queues = np.zeros(4)
        
        self.emergency_active = False
        self.emergency_direction = -1
        
        self.cumulative_arrived = 0
        
        self.reward_calc.reset()
        
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action: np.ndarray):
        """
        Execute one complete cycle (all 4 phases).
        
        Args:
            action: [duration_idx_0, duration_idx_1, duration_idx_2, duration_idx_3]
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Convert action indices to actual durations
        phase_durations = [self.green_durations[i] for i in action]
        
        # Accumulate rewards from all phases
        total_reward = 0.0
        
        # Execute all 4 phases in order
        for phase_idx in range(self.n_phases):
            self.current_phase = phase_idx
            self.phase_timer = 0
            
            duration = phase_durations[phase_idx]
            
            # Set SUMO phase
            sumo_phase = self.sumo_phases[phase_idx]
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, sumo_phase)
            
            # Execute green phase
            for _ in range(duration):
                traci.simulationStep()
                self.cumulative_arrived += traci.simulation.getArrivedNumber()
                self.simulation_step += 1
                self.phase_timer += 1
                
                # Check for emergency vehicles
                self._check_emergency_vehicles()
            
            # FIXED: All-red clearance interval
            # REMOVED incorrect phase switching - let SUMO handle transitions
            for _ in range(ALL_RED):
                traci.simulationStep()
                self.cumulative_arrived += traci.simulation.getArrivedNumber()
                self.simulation_step += 1
            
            # Get traffic state after this phase
            state = self._get_traffic_state()
            
            # Calculate reward for this phase
            reward = self.reward_calc.calculate_reward(
                queues=state["queues"],
                wait_times=state["wait_times"],
                arrived_count=state["arrived_count"],
                emergency_active=self.emergency_active,
                emergency_direction=self.emergency_direction,
                current_phase=self._map_to_2phase(phase_idx)
            )
            
            total_reward += reward
        
        # FIXED: Average reward per cycle (not sum)
        # This keeps rewards in reasonable range for PPO
        avg_reward = total_reward / self.n_phases
        
        self.cycle_count += 1
        self.current_step += 1
        self.episode_reward += avg_reward
        
        # Get final observation
        obs = self._get_observation()
        
        # Check termination
        terminated = self.simulation_step >= self.episode_length
        truncated = False
        
        info = self._get_info()
        
        return obs, avg_reward, terminated, truncated, info
    
    def _map_to_2phase(self, phase):
        """
        Map 4-phase to 2-phase for reward calculation.
        
        Args:
            phase: 4-phase index (0-3)
        
        Returns:
            2-phase index (0=EW, 1=NS)
        """
        if phase in [0, 1]:  # EW through + EW left
            return 0  # EW
        else:                # NS through + NS left
            return 1  # NS
    
    def _get_traffic_state(self):
        """Get current traffic state from SUMO."""
        # Lane configuration
        # IMPORTANT: These MUST match your SUMO network file!
        lanes = {
            'N': ['N1_to_J1_0', 'N1_to_J1_1'],
            'S': ['S1_to_J1_0', 'S1_to_J1_1'],
            'E': ['E1_to_J1_0', 'E1_to_J1_1'],
            'W': ['W1_to_J1_0', 'W1_to_J1_1']
        }
        
        queues = []
        waits = []
        
        for direction in ["N", "S", "E", "W"]:
            # Queue length (halting vehicles)
            q = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes[direction])
            queues.append(q)
            
            # Average wait time
            vehicles = []
            for lane in lanes[direction]:
                vehicles += traci.lane.getLastStepVehicleIDs(lane)
            
            if vehicles:
                w = np.mean([traci.vehicle.getWaitingTime(v) for v in vehicles])
            else:
                w = 0.0
            
            waits.append(w)
        
        return {
            "queues": queues,
            "wait_times": waits,
            "arrived_count": self.cumulative_arrived
        }
    
    def _get_observation(self):
        """
        Get normalized observation.
        
        Returns:
            np.ndarray(14,): [queues(4), delta(4), waits(4), emergency(1), phase(1)]
        """
        state = self._get_traffic_state()
        
        queues = np.array(state["queues"], dtype=np.float32)
        waits = np.array(state["wait_times"], dtype=np.float32)
        
        # Queue delta (change since last observation)
        delta = queues - self.prev_queues
        self.prev_queues = queues.copy()
        
        # Normalize
        queues = np.clip(queues / MAX_QUEUE, 0, 1)
        delta = np.clip(delta / MAX_QUEUE, -1, 1)
        waits = np.clip(waits / MAX_WAIT, 0, 1)
        
        # Combine into observation
        obs = np.concatenate([
            queues,                              # 4 features
            delta,                               # 4 features
            waits,                               # 4 features
            [float(self.emergency_active)],      # 1 feature
            [self.current_phase / 3.0]           # 1 feature (normalized to [0,1])
        ])
        
        return obs.astype(np.float32)
    
    def _check_emergency_vehicles(self):
        """Check for emergency vehicles and determine direction."""
        vehicles = traci.vehicle.getIDList()
        
        # Find emergency vehicles (route_generator uses "ambulance_",
        # admin console injects "emergency_")
        emerg = [v for v in vehicles
                 if v.startswith("ambulance_") or v.startswith("emergency_")]
        
        if emerg:
            self.emergency_active = True
            
            # Get lane of first emergency vehicle
            lane = traci.vehicle.getLaneID(emerg[0])
            
            # FIXED: More robust direction detection
            if "N1_to_J1" in lane or lane.startswith("N"):
                self.emergency_direction = 0
            elif "S1_to_J1" in lane or lane.startswith("S"):
                self.emergency_direction = 1
            elif "E1_to_J1" in lane or lane.startswith("E"):
                self.emergency_direction = 2
            elif "W1_to_J1" in lane or lane.startswith("W"):
                self.emergency_direction = 3
            else:
                # Fallback: check first character
                if lane and lane[0] in ['N', 'S', 'E', 'W']:
                    self.emergency_direction = {'N': 0, 'S': 1, 'E': 2, 'W': 3}[lane[0]]
                else:
                    print(f"  Unknown emergency vehicle lane: {lane}")
                    self.emergency_direction = -1
        
        else:
            self.emergency_active = False
            self.emergency_direction = -1
    
    def _verify_lane_names(self):
        """
        Verify that lane names in code match SUMO network.
        
        Prints warnings if lanes don't exist.
        """
        print("\n Verifying SUMO lane names...")
        
        expected_lanes = [
            'N1_to_J1_0', 'N1_to_J1_1',
            'S1_to_J1_0', 'S1_to_J1_1',
            'E1_to_J1_0', 'E1_to_J1_1',
            'W1_to_J1_0', 'W1_to_J1_1'
        ]
        
        actual_lanes = traci.lane.getIDList()
        
        missing = [l for l in expected_lanes if l not in actual_lanes]
        
        if missing:
            print(f"  WARNING: Expected lanes not found: {missing}")
            print(f"   Available lanes: {actual_lanes[:10]}...")  # Show first 10
            print(f"   You may need to update lane names in _get_traffic_state()")
        else:
            print(f" All lane names verified!")
    
    def _get_info(self):
        """Get episode info dict."""
        state = self._get_traffic_state()
        
        return {
            "queues": state["queues"],
            "wait_times": state["wait_times"],
            "arrived": state["arrived_count"],
            "cycle": self.cycle_count,
            "sim_time": self.simulation_step,
            "emergency_active": self.emergency_active,
            "emergency_direction": self.emergency_direction,
            "max_wait_per_direction": {
                "N": state["wait_times"][0],
                "S": state["wait_times"][1],
                "E": state["wait_times"][2],
                "W": state["wait_times"][3]
            },
            "phase_switches": self.cycle_count * 4  # 4 phase switches per cycle

        }
    
    def _get_curriculum_scenario(self):
        """Get scenario based on curriculum stage."""
        if self.curriculum_stage == 0:
            return "WEEKEND"
        elif self.curriculum_stage == 1:
            return "EVENING_RUSH"
        else:
            return "MORNING_RUSH"
    
    def set_curriculum_stage(self, stage: int):
        """
        Set curriculum stage (called by CurriculumCallback via env_method).
        
        Args:
            stage: Curriculum difficulty (0=WEEKEND, 1=EVENING_RUSH, 2=MORNING_RUSH)
        """
        self.curriculum_stage = max(0, min(2, stage))
        self.reward_calc.set_curriculum_stage(stage)

    def close(self):
        """Clean up SUMO simulation."""
        self.sumo.close()
    
    def render(self):
        """Render is handled by SUMO GUI if use_gui=True."""
        pass


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("4-PHASE TRAFFIC ENVIRONMENT TEST")
    print("=" * 70)
    
    env = TrafficEnv4Phase(use_gui=True)
    
    obs, info = env.reset()
    
    print(f"\n Observation shape: {obs.shape}")
    print(f"   Expected: (14,)")
    print(f"    MATCH!" if obs.shape == (14,) else "    MISMATCH!")
    
    print(f"\n Action space: {env.action_space}")
    print(f"   4 phases  8 duration options")
    
    print(f"\n Testing one cycle...")
    
    # Random action: [duration_idx for each of 4 phases]
    action = env.action_space.sample()
    
    print(f"   Action: {action}")
    print(f"   Durations: {[env.green_durations[i] for i in action]}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n Results:")
    print(f"   Reward: {reward:.2f}")
    print(f"   Queues: {info['queues']}")
    print(f"   Wait times: {[f'{w:.1f}s' for w in info['wait_times']]}")
    print(f"   Cycle: {info['cycle']}")
    
    env.close()
    
    print("\n" + "=" * 70)
    print(" TEST COMPLETE")
    print("=" * 70)