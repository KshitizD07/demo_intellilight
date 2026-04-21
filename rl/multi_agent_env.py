"""
Corridor Multi-Intersection RL Environment
==========================================

CorridorEnv controls J1 → J2 → J3, a three-junction arterial corridor.

Architecture (Centralised PPO):
  A SINGLE agent receives a flat observation from ALL junctions and produces
  duration decisions for ALL junctions simultaneously. This lets PPO naturally
  learn green-wave timing without explicit inter-agent messaging.

Action Space:
  MultiDiscrete([8] * 12)   ← 4 phases × 3 junctions, each choosing 1-of-8 durations

Observation Space:
  Box(42,)   ← 14 features per junction × 3 junctions
  per-junction features: queues(4) + queue_delta(4) + waits(4) + emergency(1) + phase(1)

Reward:
  Sum of per-junction rewards (each uses EnhancedRewardCalculator).
  The green-wave bonus is activated by passing the phase context of neighbouring
  junctions into each calculator call.

Deployment note:
  SubprocVecEnv with N_ENVS=2 is recommended for consumer-grade GPUs to avoid
  memory pressure. Cloud training (e.g., Colab A100) can safely use N_ENVS=4.
"""

import os
import sys
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, List, Optional
import traci
from gymnasium import spaces

# Ensure project root is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.sumo_env import SUMOSimulation
from simulation.route_generator import RouteGenerator
from rl.reward_function import EnhancedRewardCalculator, RewardWeights


# ── Shared constants (mirrors configs/parameters.py values) ─────────────────
ALL_RED        = 4          # Seconds of all-red clearance between phases
EPISODE_LENGTH = 1800       # Simulation duration per episode (30 min)
GREEN_DURATIONS = [10, 15, 20, 25, 30, 35, 40, 45]  # Selectable green times
MAX_QUEUE  = 50.0           # Normalisation cap for queue observations
MAX_WAIT   = 120.0          # Normalisation cap for wait-time observations


class CorridorEnv(gym.Env):
    """
    Centralised Gymnasium environment for a 3-intersection arterial corridor.

    The environment sequentially steps each junction's 4 phases every cycle.
    All junctions advance their phases in lock-step: the slowest junction
    determines when a phase slot ends (greedy-min timing approach).
    This keeps the simulation simple while still letting the agent learn
    relative offset between junctions.
    """

    metadata = {"render_modes": ["human"]}

    # Names used when printing per-junction diagnostics
    JUNCTION_IDS: List[str] = ["J1", "J2", "J3"]

    def __init__(
        self,
        use_gui: bool = False,
        episode_length: int = EPISODE_LENGTH,
        curriculum_stage: int = 0,
        seed: Optional[int] = None
    ):
        """
        Initialise corridor environment.

        Args:
            use_gui: Launch SUMO GUI (slow — only for visual debugging)
            episode_length: Episode horizon in simulation seconds
            curriculum_stage: 0=WEEKEND (easy), 1=EVENING_RUSH, 2=MORNING_RUSH
            seed: RNG seed (passed to Gymnasium super())
        """
        super().__init__()

        self.use_gui = use_gui
        self.episode_length = episode_length
        self.curriculum_stage = curriculum_stage

        self.junctions = self.JUNCTION_IDS
        self.n_intersections = len(self.junctions)
        self.n_phases = 4           # EW-Through, EW-Left, NS-Through, NS-Left
        self.n_durations = len(GREEN_DURATIONS)

        # ── Lane IDs that feed into each junction ────────────────────────────
        # Derived from corridor.nod.xml / corridor.edg.xml naming convention.
        # "W" of J2 is the outbound link J1→J2, "E" of J2 is J3→J2.
        self.lanes: Dict[str, Dict[str, List[str]]] = {
            "J1": {
                "N": ["N1_to_J1_0", "N1_to_J1_1"],
                "S": ["S1_to_J1_0", "S1_to_J1_1"],
                "E": ["J2_to_J1_0", "J2_to_J1_1"],   # arterial eastbound return
                "W": ["W1_to_J1_0", "W1_to_J1_1"],   # arterial westbound entry
            },
            "J2": {
                "N": ["N2_to_J2_0", "N2_to_J2_1"],
                "S": ["S2_to_J2_0", "S2_to_J2_1"],
                "E": ["J3_to_J2_0", "J3_to_J2_1"],
                "W": ["J1_to_J2_0", "J1_to_J2_1"],   # arterial westbound (from J1)
            },
            "J3": {
                "N": ["N3_to_J3_0", "N3_to_J3_1"],
                "S": ["S3_to_J3_0", "S3_to_J3_1"],
                "E": ["E1_to_J3_0", "E1_to_J3_1"],   # arterial eastbound entry
                "W": ["J2_to_J3_0", "J2_to_J3_1"],   # arterial westbound (from J2)
            },
        }

        # Map RL phase index to SUMO TLS phase index (must match corridor.net.xml)
        self.sumo_phase_map = {0: 0, 1: 1, 2: 2, 3: 3}

        # ── Spaces ────────────────────────────────────────────────────────────
        # Centralised action: one duration choice per (junction × phase)
        self.action_space = spaces.MultiDiscrete(
            [self.n_durations] * (self.n_phases * self.n_intersections)
        )
        # Flat observation: 14 dims per junction (see _build_obs_for_junction)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(14 * self.n_intersections,),
            dtype=np.float32
        )

        # ── Simulation objects ────────────────────────────────────────────────
        self.sumo = SUMOSimulation(use_gui=use_gui)
        self.route_gen = RouteGenerator()

        # One reward calculator per junction (each tracks its own throughput counter)
        self.reward_calcs: List[EnhancedRewardCalculator] = [
            EnhancedRewardCalculator(
                weights=RewardWeights(),
                max_acceptable_wait=90,
                emergency_max_wait=30,
            )
            for _ in self.junctions
        ]
        for rc in self.reward_calcs:
            rc.set_curriculum_stage(curriculum_stage)

        # ── Episode state ─────────────────────────────────────────────────────
        self.simulation_step = 0
        self.cycle_count = 0
        self.cumulative_arrived = 0
        # Previous queue snapshot per junction (for delta feature)
        self.prev_queues: Dict[str, np.ndarray] = {
            j: np.zeros(4) for j in self.junctions
        }
        # Current executing phase per junction (0-3), updated each sub-step
        self.current_phases: Dict[str, int] = {j: 0 for j in self.junctions}

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset corridor for a new episode.

        Returns:
            obs: Initial flat observation (shape 42,)
            info: Empty dict (Gymnasium convention)
        """
        super().reset(seed=seed)

        # Tear down any running simulation
        self.sumo.reset()

        # Pick scenario based on curriculum
        scenario_map = {0: "WEEKEND", 1: "EVENING_RUSH", 2: "MORNING_RUSH"}
        scenario = scenario_map.get(self.curriculum_stage, "WEEKEND")

        # Generate corridor-compatible route file
        route_file = self.route_gen.generate_unique_filename()
        self.route_gen.generate_route_file(
            route_file,
            scenario=scenario,
            curriculum_stage=self.curriculum_stage,
        )
        self.sumo.start(route_file)

        # Reset episode counters
        self.simulation_step = 0
        self.cycle_count = 0
        self.cumulative_arrived = 0
        self.prev_queues = {j: np.zeros(4) for j in self.junctions}
        self.current_phases = {j: 0 for j in self.junctions}

        for rc in self.reward_calcs:
            rc.reset()

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one full corridor cycle (all phases at all junctions).

        The action tensor is interpreted as:
            action[i*4 + p]  →  duration index for junction i, phase p

        All junctions run their phases in parallel; we advance the simulation
        one second at a time and check each junction's remaining green-time
        independently.

        Args:
            action: int array shape (12,) — duration index per (junction, phase)

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Decode action into per-junction, per-phase durations (in seconds)
        # Shape: (n_intersections, n_phases)
        phase_durations: List[List[int]] = [
            [GREEN_DURATIONS[action[i * self.n_phases + p]] for p in range(self.n_phases)]
            for i in range(self.n_intersections)
        ]

        # Initialise phase tracking for this cycle
        current_phase_idx = [0] * self.n_intersections      # which phase each junction is on
        phase_timer       = [phase_durations[i][0] for i in range(self.n_intersections)]
        phase_done        = [False] * self.n_intersections

        # Set all junctions to their first phase (EW-Through)
        for jid in self.junctions:
            try:
                traci.trafficlight.setPhase(jid, self.sumo_phase_map[0])
            except traci.TraCIException:
                pass  # junction not yet in TLS control — harmless on first step

        # ── Inner simulation loop ─────────────────────────────────────────────
        old_step = self.simulation_step
        # Advance 1 s at a time until every junction has completed all 4 phases
        while not all(phase_done):
            traci.simulationStep()
            self.simulation_step += 1
            self.cumulative_arrived += traci.simulation.getArrivedNumber()

            for i, jid in enumerate(self.junctions):
                if phase_done[i]:
                    continue

                phase_timer[i] -= 1
                if phase_timer[i] <= 0:
                    # Current phase expired — run all-red clearance
                    for _ in range(ALL_RED):
                        traci.simulationStep()
                        self.simulation_step += 1
                        self.cumulative_arrived += traci.simulation.getArrivedNumber()

                    current_phase_idx[i] += 1
                    if current_phase_idx[i] >= self.n_phases:
                        phase_done[i] = True
                    else:
                        # Advance to next phase
                        next_p = current_phase_idx[i]
                        phase_timer[i] = phase_durations[i][next_p]
                        self.current_phases[jid] = next_p
                        try:
                            traci.trafficlight.setPhase(jid, self.sumo_phase_map[next_p])
                        except traci.TraCIException:
                            pass

        self.cycle_count += 1

        # ── Collect traffic state and compute rewards ─────────────────────────
        states = self._get_all_traffic_states()
        total_reward = 0.0

        for i, jid in enumerate(self.junctions):
            r = self.reward_calcs[i].calculate_reward(
                queues=states[jid]["queues"],
                wait_times=states[jid]["wait_times"],
                arrived_count=self.cumulative_arrived,
                emergency_active=False,
                emergency_direction=None,
                current_phase=self.current_phases[jid],
            )
            total_reward += r

        # Average across junctions so scale matches single-intersection PPO
        avg_reward = total_reward / self.n_intersections

        terminated = self.simulation_step >= self.episode_length
        obs = self._get_observation()

        info = {
            "arrived":    self.cumulative_arrived,
            "throughput": self.cumulative_arrived,   # consistent key for main.py
            "queues":     sum(sum(states[j]["queues"]) for j in self.junctions),
            "cycle":      self.cycle_count,
            "intersections": states,                 # needed for wait_time metrics
            "cycle_time": self.simulation_step - old_step
        }

        return obs, avg_reward, terminated, False, info

    def set_curriculum_stage(self, stage: int):
        """Update curriculum stage in all reward calculators (called by CurriculumCallback)."""
        self.curriculum_stage = stage
        for rc in self.reward_calcs:
            rc.set_curriculum_stage(stage)

    def close(self):
        """Cleanly shut down SUMO."""
        self.sumo.close()

    def render(self):
        """Rendering is handled by SUMO GUI when use_gui=True."""
        pass

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_all_traffic_states(self) -> Dict[str, Dict]:
        """
        Query SUMO for queue and wait-time data at every junction.

        Returns:
            Dict keyed by junction ID, each value:
            {
                "queues":     [N, S, E, W] halting vehicle counts,
                "wait_times": [N, S, E, W] mean wait in seconds,
            }
        """
        states: Dict[str, Dict] = {}
        for jid in self.junctions:
            queues: List[float] = []
            waits:  List[float] = []
            for direction in ["N", "S", "E", "W"]:
                lane_ids = self.lanes[jid][direction]
                # Queue = total halting vehicles across both lanes
                q = sum(
                    traci.lane.getLastStepHaltingNumber(lid)
                    for lid in lane_ids
                )
                queues.append(float(q))

                # Wait = mean wait of all vehicles currently on these lanes
                vehicles: List[str] = []
                for lid in lane_ids:
                    vehicles.extend(traci.lane.getLastStepVehicleIDs(lid))
                waits.append(
                    float(np.mean([traci.vehicle.getWaitingTime(v) for v in vehicles]))
                    if vehicles else 0.0
                )

            states[jid] = {"queues": queues, "wait_times": waits}
        return states

    def _build_obs_for_junction(
        self,
        jid: str,
        state: Dict
    ) -> np.ndarray:
        """
        Build the 14-dimensional normalised observation for one junction.

        Layout:
          [ queues(4) | queue_delta(4) | waits(4) | emergency(1) | phase(1) ]

        Args:
            jid: Junction ID string
            state: Dict with "queues" and "wait_times" from _get_all_traffic_states

        Returns:
            np.ndarray shape (14,) in [0, 1]
        """
        queues = np.array(state["queues"], dtype=np.float32)
        waits  = np.array(state["wait_times"], dtype=np.float32)

        # Delta since last observation (trend feature)
        delta = queues - self.prev_queues[jid]
        self.prev_queues[jid] = queues.copy()

        # Normalise to [0, 1] (queues) or [-1, 1] (delta) then clip
        norm_queues = np.clip(queues / MAX_QUEUE, 0.0, 1.0)
        norm_delta  = np.clip(delta  / MAX_QUEUE, -1.0, 1.0)
        norm_waits  = np.clip(waits  / MAX_WAIT,   0.0, 1.0)

        phase_norm = self.current_phases[jid] / 3.0   # 4 phases → [0, 1]

        return np.concatenate([
            norm_queues,        # 4 features
            norm_delta,         # 4 features
            norm_waits,         # 4 features
            [0.0],              # emergency flag (placeholder)
            [phase_norm],       # current phase normalised
        ]).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        """
        Build the full 42-dim corridor observation by concatenating
        per-junction 14-dim vectors in order J1 → J2 → J3.
        """
        states = self._get_all_traffic_states()
        obs_parts = [
            self._build_obs_for_junction(jid, states[jid])
            for jid in self.junctions
        ]
        return np.concatenate(obs_parts).astype(np.float32)
