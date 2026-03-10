"""
IntelliLight Traffic Environment Module
========================================

Main RL environment for training traffic signal control agents.

This environment:
- Connects to SUMO via sumo_env.py
- Generates traffic scenarios via route_generator.py
- Calculates rewards via reward_function.py
- Implements the Gymnasium Env interface for RL training

This is the "heart" of the system that ties everything together.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import logging
from typing import Optional, Dict, Tuple, Any

from simulation.sumo_env import SUMOSimulation
from simulation.route_generator import RouteGenerator
from rl.reward_function import RewardCalculator
from configs.parameters import (
    NetworkTopology, SignalTiming, ObservationConfig,
    ActionConfig, EpisodeConfig, CurriculumConfig,
    ResourceConfig, Paths
)

logger = logging.getLogger(__name__)


class TrafficEnv(gym.Env):
    """
    Gymnasium environment for traffic signal control.
    
    This environment allows an RL agent to control traffic lights
    at an intersection to optimize traffic flow.
    
    Observation Space:
        Box(9,) normalized to [0,1]:
        - [0:4]: Queue lengths for W, E, N, S
        - [4:8]: Waiting times for W, E, N, S  
        - [8]: Emergency vehicle flag
    
    Action Space:
        MultiDiscrete([2, 3]):
        - [0]: Direction (0=EW, 1=NS)
        - [1]: Duration (0=10s, 1=20s, 2=30s)
    
    Reward:
        Multi-objective combining throughput, efficiency, fairness,
        wait time, queue length, starvation, and emergency response.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        use_gui: bool = False,
        curriculum_stage: int = 0,
        sumo_cfg: Optional[str] = None,
       
    ):
        """
        Initialize the traffic environment.
        
        Args:
            use_gui: Whether to render SUMO GUI for visualization
            curriculum_stage: Current curriculum learning stage (0-2)
            sumo_cfg: Path to SUMO config file (defaults to config)
        """
        super().__init__()
        
        logger.info(
            f"Initializing TrafficEnv: GUI={use_gui}, "
            f"curriculum_stage={curriculum_stage}"
        )
        
        # Configuration
        self.use_gui = use_gui
        self.curriculum_stage = curriculum_stage
        self.episode_length = EpisodeConfig.LENGTH
        
        # Define Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=ObservationConfig.SHAPE,
            dtype=np.float32
        )
        
        self.action_space = spaces.MultiDiscrete([
            ActionConfig.N_DIRECTIONS,
            len(SignalTiming.GREEN_OPTIONS)
        ])
        
        # Initialize components
        self.sumo = SUMOSimulation(
            sumo_cfg=sumo_cfg,
            use_gui=use_gui
        )
        
        self.route_gen = RouteGenerator()
        
        self.reward_calc = RewardCalculator(
            curriculum_stage=curriculum_stage
        )
        
        # Episode tracking
        self.current_step = 0
        self.simulation_step = 0
        self.episode_count = 0
        self.current_route_file = None
        
        # Performance metrics
        self.episode_metrics = {
            'total_reward': 0.0,
            'total_wait': 0.0,
            'total_throughput': 0,
            'emergency_count': 0
        }
        
        logger.info("TrafficEnv initialized successfully")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
        
        Returns:
            tuple: (observation, info)
                - observation: Initial state
                - info: Episode information
        """
        # Gymnasium requires calling super().reset()
        super().reset(seed=seed)
        
        self.episode_count += 1
        self.current_step = 0
        self.simulation_step = 0
        
        logger.info(f"Starting episode {self.episode_count}")
        
        # Periodic cleanup
        if self.episode_count % ResourceConfig.CLEANUP_INTERVAL == 0:
            cleaned = self.route_gen.cleanup_old_routes()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} old route files")
        
        # Generate new traffic scenario
        self.current_route_file = self.route_gen.generate_unique_filename()
        scenario_info = self.route_gen.generate_route_file(
            self.current_route_file,
            scenario="RANDOM",
            curriculum_stage=self.curriculum_stage
        )
        
        logger.info(
            f"Episode {self.episode_count}: {scenario_info['scenario']} "
            f"({scenario_info['total_flow']} veh/hour)"
        )
        
        # Start SUMO
        self.sumo.start(self.current_route_file)
        
        # Warm-up simulation
        for _ in range(5):
            self.sumo.step()
            self.simulation_step += 1
        
        # Reset reward calculator
        self.reward_calc.reset()
        
        # Reset metrics
        self.episode_metrics = {
            'total_reward': 0.0,
            'total_wait': 0.0,
            'total_throughput': 0,
            'emergency_count': 0
        }
        
        # Get initial observation
        obs = self._get_observation()
        
        # Prepare info dict
        info = {
            'scenario': scenario_info['scenario'],
            'traffic_flow': scenario_info['total_flow'],
            'emergency': scenario_info['emergency'] is not None
        }
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action from agent [direction, duration_idx]
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: Next state
                - reward: Reward for this step
                - terminated: Episode ended naturally
                - truncated: Episode ended due to time limit
                - info: Additional information
        """
        # Decode action
        direction = int(action[0])  # 0=EW, 1=NS
        duration_idx = int(action[1])  # Index into GREEN_OPTIONS
        duration = SignalTiming.GREEN_OPTIONS[duration_idx]
        
        # Check for emergency vehicle
        emergency_active = self._detect_emergency()
        
        if emergency_active:
            self.episode_metrics['emergency_count'] += 1
            logger.debug("Emergency vehicle detected!")
        
        # Set traffic light based on action
        self._apply_traffic_light_action(direction, duration)
        self.current_step+=1
        
        # Collect metrics before calculating reward
        queues, wait_times = self._get_traffic_metrics()
        arrived_count = self.sumo.get_arrived_vehicles()
        # if self.current_step%5==0:
        #     print(f"[Debug] Step {self.current_step}: arrived_count={arrived_count},"f"prev={self.reward_calc.previous_arrived_count}")
        # Calculate reward
        reward = self.reward_calc.calculate_reward(
            queues=queues,
            wait_times=wait_times,
            arrived_count=arrived_count,  # Total vehicles
            current_step=self.current_step,
            emergency_active=emergency_active,
            action_direction=direction
        )
        
        # Update metrics
        self.episode_metrics['total_reward'] += reward
        self.episode_metrics['total_wait'] += np.sum(wait_times)
        self.episode_metrics['total_throughput'] += arrived_count
        
        # Get next observation
        obs = self._get_observation()
        
        # Check termination
        terminated = False  # Traffic doesn't "end" naturally
        truncated = self.simulation_step >= self.episode_length
        
        # Prepare info dict
        info = {
            'step': self.current_step,
            'simulation_step': self.simulation_step,
            'reward_components': self.reward_calc.get_component_breakdown(),
            'queues': queues.tolist(),
            'wait_times': wait_times.tolist(),
            'emergency': emergency_active,
            'throughput': arrived_count
        }
        
        return obs, reward, terminated, truncated, info
    
    def _apply_traffic_light_action(self, direction: int, duration: int):
        """
        Apply traffic light action and simulate for the duration.
        
        Args:
            direction: 0=EW, 1=NS
            duration: Green light duration in seconds
        """
        # Set traffic light state (16 characters for the network)
        if direction == 0:
            # East-West green
            state = "rrrrrrrrGGGgGGGg"
        else:
            # North-South green
            state = "GGGgGGGgrrrrrrrr"
        
        self.sumo.set_traffic_light_state(
            NetworkTopology.TRAFFIC_LIGHT_ID,
            state
        )
        
        # Run simulation for green duration
        for _ in range(duration):
            self.sumo.step()
            self.simulation_step += 1
        
        # All-red safety phase
        self.sumo.set_traffic_light_state(
            NetworkTopology.TRAFFIC_LIGHT_ID,
            "rrrrrrrrrrrrrrrr"
        )
        
        for _ in range(SignalTiming.ALL_RED):
            self.sumo.step()
            self.simulation_step += 1
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current traffic state as normalized observation.
        
        Returns:
            np.ndarray: Normalized observation [0, 1]
        """
        obs = np.zeros(ObservationConfig.SHAPE, dtype=np.float32)
        
        # Get traffic metrics
        queues, wait_times = self._get_traffic_metrics()
        
        # Normalize queue lengths [0, 1]
        obs[ObservationConfig.QUEUE_INDICES] = np.clip(
            queues / ObservationConfig.MAX_CARS_ON_LANE,
            0.0, 1.0
        )
        
        # Normalize wait times [0, 1]
        obs[ObservationConfig.WAIT_INDICES] = np.clip(
            wait_times / ObservationConfig.MAX_WAIT_TIME,
            0.0, 1.0
        )
        
        # Emergency vehicle flag
        obs[ObservationConfig.EMERGENCY_INDEX] = 1.0 if self._detect_emergency() else 0.0
        
        return obs
    
    def _get_traffic_metrics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect current traffic metrics from SUMO.
        
        Returns:
            tuple: (queues, wait_times) for [W, E, N, S]
        """
        queues = np.zeros(4, dtype=np.float32)
        wait_times = np.zeros(4, dtype=np.float32)
        
        for idx, direction in enumerate(["W", "E", "N", "S"]):
            lanes = NetworkTopology.LANES[direction]
            
            # Sum vehicle counts across lanes
            for lane in lanes:
                queues[idx] += self.sumo.get_lane_vehicle_count(lane)
                wait_times[idx] += self.sumo.get_lane_waiting_time(lane)
        
        return queues, wait_times
    
    def _detect_emergency(self) -> bool:
        """
        Detect if an emergency vehicle is present in the simulation.
        
        Returns:
            bool: True if emergency vehicle detected
        """
        vehicle_ids = self.sumo.get_all_vehicle_ids()
        
        for veh_id in vehicle_ids:
            veh_type = self.sumo.get_vehicle_type(veh_id)
            if veh_type == "ambulance":
                return True
        
        return False
    
    def render(self):
        """Render the environment (SUMO GUI handles visualization)."""
        if self.use_gui:
            pass  # SUMO GUI is already rendering
        else:
            logger.warning("Render called but GUI is disabled")
    
    def close(self):
        """Clean up resources."""
        logger.info("Closing TrafficEnv")
        self.sumo.close()
        
        # Log final episode metrics if episode was run
        if self.episode_count > 0:
            logger.info(
                f"Episode {self.episode_count} summary: "
                f"reward={self.episode_metrics['total_reward']:.2f}, "
                f"throughput={self.episode_metrics['total_throughput']}"
            )
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the current episode.
        
        Returns:
            dict: Episode metrics including reward, wait time, throughput
        """
        return self.episode_metrics.copy()
    
    def set_curriculum_stage(self, stage: int):
        """
        Update curriculum learning stage.
        
        Args:
            stage: New curriculum stage (0-2)
        """
        if stage != self.curriculum_stage:
            self.curriculum_stage = stage
            self.reward_calc.set_curriculum_stage(stage)
            logger.info(f"Curriculum stage updated to {stage}")


if __name__ == "__main__":
    """Simple test of the traffic environment."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing TrafficEnv Module")
    print("=" * 60)
    
    try:
        # Create environment
        print("\n1. Creating environment...")
        env = TrafficEnv(use_gui=False, curriculum_stage=0)
        print("   ✓ Environment created")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Reset environment
        print("\n2. Resetting environment...")
        obs, info = env.reset()
        print("   ✓ Environment reset")
        print(f"   Initial observation shape: {obs.shape}")
        print(f"   Scenario: {info['scenario']}")
        print(f"   Traffic flow: {info['traffic_flow']} veh/hour")
        
        # Take some random actions
        print("\n3. Taking random actions...")
        total_reward = 0
        
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 5 == 0:
                print(f"   Step {step}: reward={reward:.2f}, throughput={info['throughput']}")
            
            if terminated or truncated:
                print(f"   Episode ended at step {step}")
                break
        
        print(f"\n   Total reward: {total_reward:.2f}")
        
        # Get episode metrics
        print("\n4. Episode metrics...")
        metrics = env.get_episode_metrics()
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        # Close environment
        print("\n5. Closing environment...")
        env.close()
        print("   ✓ Environment closed")
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)