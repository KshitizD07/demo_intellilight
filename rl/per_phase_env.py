"""
ILPS Per-Phase Corridor Environment
====================================

Independent Learning with Parameter Sharing (ILPS) environment for
multi-junction traffic signal control.

Architecture:
  A SINGLE shared policy (universal junction brain) controls every junction.
  The environment presents one junction at a time; the same neural network
  produces decisions for J1, J2, … JN without knowing which junction it
  controls.

Action Space:
  Discrete(8) — a single duration index mapping to GREEN_DURATIONS.
  One action per step, one step per junction per phase.

Observation Space:
  Box(20,) — 20-dim per-junction observation:
    [0:4]   queues  (N, S, E, W)           — normalised halting vehicles
    [4:8]   deltas  (N, S, E, W)           — queue change since last obs
    [8:12]  waits   (N, S, E, W)           — normalised mean wait times
    [12]    emergency_flag                  — emergency vehicle present
    [13]    current_phase                   — phase index / 3.0
    [14]    upstream_phase                  — neighbour phase (or 0)
    [15]    time_since_upstream_switch      — seconds since neighbour changed phase
    [16]    downstream_phase               — neighbour phase (or 0)
    [17]    time_since_downstream_switch    — seconds since neighbour changed phase
    [18]    has_upstream                    — 1.0 if upstream neighbour exists
    [19]    has_downstream                  — 1.0 if downstream neighbour exists

Execution Model (parallel per-phase):
  For each phase (0→1→2→3):
    1.  N steps collect one action per junction (no SUMO advance).
    2.  After the Nth action, execute the phase at ALL junctions
        simultaneously: set TLS, advance SUMO for max(durations),
        compute per-junction rewards.
    3.  Rewards are assigned to each junction when it is next queried.

Web-Ready Hooks:
  Every RL decision is logged to `self.decision_log` as a dict that a
  future WebSocket layer can broadcast.  The override interface is
  defined but NOT active during training (see AdminPanelConfig).

Scalability:
  Trained on 3 junctions → deploy to 9, 25, or 100+ without retraining.
  The policy never sees junction count or position — only local state
  + binary topology flags.
"""

import os
import sys
import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, List, Optional
import traci
from gymnasium import spaces
from datetime import datetime

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.sumo_env import SUMOSimulation
from simulation.route_generator import RouteGenerator
from rl.reward_function import EnhancedRewardCalculator, RewardWeights


# ── Constants (from configs/parameters.py) ───────────────────────────────────
ALL_RED         = 4
EPISODE_LENGTH  = 1800
GREEN_DURATIONS = [15, 20, 25, 30, 35, 40, 45, 50]
MAX_QUEUE       = 50.0
MAX_WAIT        = 120.0
N_DURATIONS     = len(GREEN_DURATIONS)
MAX_PHASE_TIME  = 120.0   # Normalisation cap for time_since_switch


class PerPhaseCorridorEnv(gym.Env):
    """
    ILPS Gymnasium environment for per-phase multi-junction control.

    Each ``step()`` handles ONE junction for the CURRENT phase.
    After all junctions have submitted actions for a phase, the phase is
    executed at every junction in parallel (SUMO advances once).

    This gives:
      - Per-phase decision freshness (obs updated every phase, not every cycle)
      - ILPS compatibility (shared policy, Discrete(8) action)
      - Correct reward attribution via deferred per-junction rewards
      - Realistic parallel execution (SUMO simulates all junctions together)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_junctions: int = 3,
        use_gui: bool = False,
        episode_length: int = EPISODE_LENGTH,
        curriculum_stage: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialise ILPS per-phase environment.

        Args:
            n_junctions: Number of intersections (default 3, scalable to any N)
            use_gui:     Launch SUMO GUI (slow — debugging only)
            episode_length: Episode horizon in simulation seconds
            curriculum_stage: 0=WEEKEND, 1=EVENING_RUSH, 2=MORNING_RUSH
            seed:        RNG seed
        """
        super().__init__()

        self.n_junctions = n_junctions
        self.use_gui = use_gui
        self.episode_length = episode_length
        self.curriculum_stage = curriculum_stage

        self.junction_ids: List[str] = [f"J{i+1}" for i in range(n_junctions)]
        self.n_phases = 4

        # ── Spaces ────────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(N_DURATIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )

        # ── Lane mappings (corridor naming convention) ────────────────────
        # Build lane IDs dynamically for any number of junctions
        self.lanes: Dict[str, Dict[str, List[str]]] = {}
        for i, jid in enumerate(self.junction_ids):
            self.lanes[jid] = self._build_lane_ids(i, jid)

        # SUMO phase index → SUMO TLS phase (identity for standard 4-phase)
        self.sumo_phase_map = {0: 0, 1: 1, 2: 2, 3: 3}

        # ── Simulation ────────────────────────────────────────────────────
        self.sumo = SUMOSimulation(use_gui=use_gui)
        self.route_gen = RouteGenerator()

        # One reward calculator per junction
        self.reward_calcs: List[EnhancedRewardCalculator] = [
            EnhancedRewardCalculator(
                weights=RewardWeights(),
                max_acceptable_wait=90,
                emergency_max_wait=30,
            )
            for _ in self.junction_ids
        ]
        for rc in self.reward_calcs:
            rc.set_curriculum_stage(curriculum_stage)

        # ── Episode state ─────────────────────────────────────────────────
        self.simulation_step = 0
        self.cycle_count = 0
        self.cumulative_arrived = 0

        # Per-junction previous queues (for delta feature)
        self.prev_queues: Dict[str, np.ndarray] = {
            j: np.zeros(4) for j in self.junction_ids
        }

        # Phase tracking
        self.current_phase_idx = 0          # Which phase we're deciding (0-3)
        self.current_junction_idx = 0       # Which junction within the phase (0..N-1)

        # Phase timing for neighbour coordination
        self.current_phases: Dict[str, int] = {j: 0 for j in self.junction_ids}
        self.phase_switch_time: Dict[str, float] = {
            j: 0.0 for j in self.junction_ids
        }

        # Batch-decision buffer: actions collected before parallel execution
        self._phase_actions: List[Optional[int]] = [None] * n_junctions

        # Deferred rewards: computed during execution, returned on next query
        self._pending_rewards: List[float] = [0.0] * n_junctions

        # ── Web-ready hooks ───────────────────────────────────────────────
        # Decision log: list of dicts, each describing an RL decision.
        # A future WebSocket server reads this list and broadcasts.
        self.decision_log: List[Dict] = []

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        self.sumo.reset()

        # Pick scenario via curriculum
        scenario_map = {0: "WEEKEND", 1: "EVENING_RUSH", 2: "MORNING_RUSH"}
        scenario = scenario_map.get(self.curriculum_stage, "WEEKEND")

        route_file = self.route_gen.generate_unique_filename()
        self.route_gen.generate_route_file(
            route_file,
            scenario=scenario,
            curriculum_stage=self.curriculum_stage,
        )
        self.sumo.start(route_file)

        # Reset all state
        self.simulation_step = 0
        self.cycle_count = 0
        self.cumulative_arrived = 0
        self.prev_queues = {j: np.zeros(4) for j in self.junction_ids}
        self.current_phase_idx = 0
        self.current_junction_idx = 0
        self.current_phases = {j: 0 for j in self.junction_ids}
        self.phase_switch_time = {j: 0.0 for j in self.junction_ids}
        self._phase_actions = [None] * self.n_junctions
        self._pending_rewards = [0.0] * self.n_junctions
        self.decision_log.clear()

        for rc in self.reward_calcs:
            rc.reset()

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Process one junction's phase-duration decision.

        If this is the last junction for the current phase, execute the
        phase at ALL junctions simultaneously in SUMO.

        Args:
            action: Duration index (0-7) → GREEN_DURATIONS[action]

        Returns:
            obs, reward, terminated, truncated, info
        """
        ji = self.current_junction_idx
        junction_id = self.junction_ids[ji]
        duration = GREEN_DURATIONS[action]

        # Store this junction's decision
        self._phase_actions[ji] = action

        # Log decision for web-ready hook
        self._log_decision(junction_id, self.current_phase_idx, duration)

        # Retrieve this junction's pending reward (from previous execution)
        reward = self._pending_rewards[ji]
        self._pending_rewards[ji] = 0.0

        # Check if all junctions decided for this phase
        is_last_junction = (ji == self.n_junctions - 1)

        if is_last_junction:
            # ── PARALLEL EXECUTION ────────────────────────────────────
            # All junctions decided → execute this phase simultaneously
            execution_rewards = self._execute_phase_parallel()

            # The LAST junction gets its reward immediately
            reward = execution_rewards[ji]

            # Store rewards for other junctions (returned on their next step)
            for k in range(self.n_junctions - 1):
                self._pending_rewards[k] = execution_rewards[k]

            # Advance to next phase
            self.current_junction_idx = 0
            self.current_phase_idx = (self.current_phase_idx + 1) % self.n_phases
            if self.current_phase_idx == 0:
                self.cycle_count += 1

            # Clear decision buffer
            self._phase_actions = [None] * self.n_junctions
        else:
            # More junctions to decide — advance junction pointer
            self.current_junction_idx += 1

        # Build observation for the NEXT junction to act
        obs = self._get_observation()

        terminated = self.simulation_step >= self.episode_length
        truncated = False

        info = {
            "junction_id": junction_id,
            "phase": self.current_phase_idx,
            "duration": duration,
            "cycle": self.cycle_count,
            "sim_time": self.simulation_step,
            "arrived": self.cumulative_arrived,
            "throughput": self.cumulative_arrived,
            "is_execution_step": is_last_junction,
        }

        return obs, reward, terminated, truncated, info

    def set_curriculum_stage(self, stage: int):
        """Update curriculum stage for all reward calculators."""
        self.curriculum_stage = stage
        for rc in self.reward_calcs:
            rc.set_curriculum_stage(stage)

    def close(self):
        """Shut down SUMO cleanly."""
        self.sumo.close()

    def render(self):
        """Rendering handled by SUMO GUI when use_gui=True."""
        pass

    # ── Parallel Phase Execution ──────────────────────────────────────────

    def _execute_phase_parallel(self) -> List[float]:
        """
        Execute the current phase at ALL junctions simultaneously.

        1.  Set every junction's TLS to the current phase.
        2.  Advance SUMO for max(durations) seconds.
        3.  Run all-red clearance.
        4.  Compute per-junction rewards.

        Returns:
            List of per-junction reward floats.
        """
        # Decode durations from stored actions
        durations = [
            GREEN_DURATIONS[self._phase_actions[i]]
            for i in range(self.n_junctions)
        ]
        max_duration = max(durations)

        # Set all junctions to the current phase
        sumo_phase = self.sumo_phase_map[self.current_phase_idx]
        for jid in self.junction_ids:
            try:
                traci.trafficlight.setPhase(jid, sumo_phase)
            except traci.TraCIException:
                pass

        # Update phase tracking
        for jid in self.junction_ids:
            self.current_phases[jid] = self.current_phase_idx
            self.phase_switch_time[jid] = float(self.simulation_step)

        # Advance SUMO for max_duration seconds
        for _ in range(max_duration):
            traci.simulationStep()
            self.simulation_step += 1
            self.cumulative_arrived += traci.simulation.getArrivedNumber()

        # All-red clearance
        for _ in range(ALL_RED):
            traci.simulationStep()
            self.simulation_step += 1
            self.cumulative_arrived += traci.simulation.getArrivedNumber()

        # Compute per-junction rewards
        states = self._get_all_traffic_states()
        rewards: List[float] = []

        for i, jid in enumerate(self.junction_ids):
            r = self.reward_calcs[i].calculate_reward(
                queues=states[jid]["queues"],
                wait_times=states[jid]["wait_times"],
                arrived_count=self.cumulative_arrived,
                emergency_active=False,
                emergency_direction=None,
                current_phase=self.current_phase_idx,
            )
            rewards.append(r)

        return rewards

    # ── Observation Building ──────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Build the 20-dim observation for the current junction.

        Layout:
          [0:4]  norm_queues     [4:8]  norm_deltas    [8:12] norm_waits
          [12]   emergency       [13]   phase_norm
          [14]   upstream_phase  [15]   upstream_time
          [16]   downstream_phase [17]  downstream_time
          [18]   has_upstream    [19]   has_downstream
        """
        ji = self.current_junction_idx
        jid = self.junction_ids[ji]

        # ── Local features (14 dims) ──────────────────────────────────
        state = self._get_junction_traffic_state(jid)

        queues = np.array(state["queues"], dtype=np.float32)
        waits = np.array(state["wait_times"], dtype=np.float32)

        delta = queues - self.prev_queues[jid]
        self.prev_queues[jid] = queues.copy()

        norm_queues = np.clip(queues / MAX_QUEUE, 0.0, 1.0)
        norm_delta = np.clip(delta / MAX_QUEUE, -1.0, 1.0)
        norm_waits = np.clip(waits / MAX_WAIT, 0.0, 1.0)

        emergency = 0.0  # Placeholder — populated by _check_emergency
        phase_norm = self.current_phase_idx / 3.0

        # ── Neighbour features (4 dims) ───────────────────────────────
        up_state = self._get_neighbor_state(ji - 1)
        dn_state = self._get_neighbor_state(ji + 1)

        # ── Topology features (2 dims) ────────────────────────────────
        has_upstream = 1.0 if ji > 0 else 0.0
        has_downstream = 1.0 if ji < self.n_junctions - 1 else 0.0

        obs = np.array([
            *norm_queues,                     # 4
            *norm_delta,                      # 4
            *norm_waits,                      # 4
            emergency,                        # 1
            phase_norm,                       # 1
            up_state["phase"],                # 1
            up_state["time_since_switch"],    # 1
            dn_state["phase"],                # 1
            dn_state["time_since_switch"],    # 1
            has_upstream,                     # 1
            has_downstream,                   # 1
        ], dtype=np.float32)                  # Total: 20

        return obs

    def _get_neighbor_state(self, neighbor_idx: int) -> Dict[str, float]:
        """
        Get normalised phase state of a neighbouring junction.

        Returns sentinel zeros when the neighbour doesn't exist
        (boundary junctions).
        """
        if neighbor_idx < 0 or neighbor_idx >= self.n_junctions:
            return {"phase": 0.0, "time_since_switch": 0.0}

        neighbor_id = self.junction_ids[neighbor_idx]
        phase_norm = self.current_phases[neighbor_id] / 3.0

        elapsed = float(self.simulation_step) - self.phase_switch_time[neighbor_id]
        time_norm = np.clip(elapsed / MAX_PHASE_TIME, 0.0, 1.0)

        return {"phase": phase_norm, "time_since_switch": time_norm}

    # ── Traffic State Queries ─────────────────────────────────────────────

    def _get_all_traffic_states(self) -> Dict[str, Dict]:
        """Query SUMO for queue and wait-time data at every junction."""
        return {jid: self._get_junction_traffic_state(jid) for jid in self.junction_ids}

    def _get_junction_traffic_state(self, jid: str) -> Dict:
        """
        Get traffic state for a single junction.

        Returns:
            {"queues": [N,S,E,W], "wait_times": [N,S,E,W]}
        """
        queues: List[float] = []
        waits: List[float] = []

        for direction in ["N", "S", "E", "W"]:
            lane_ids = self.lanes[jid][direction]

            q = sum(
                traci.lane.getLastStepHaltingNumber(lid)
                for lid in lane_ids
            )
            queues.append(float(q))

            vehicles: List[str] = []
            for lid in lane_ids:
                vehicles.extend(traci.lane.getLastStepVehicleIDs(lid))

            waits.append(
                float(np.mean([traci.vehicle.getWaitingTime(v) for v in vehicles]))
                if vehicles else 0.0
            )

        return {"queues": queues, "wait_times": waits}

    # ── Lane ID Builder ───────────────────────────────────────────────────

    def _build_lane_ids(self, idx: int, jid: str) -> Dict[str, List[str]]:
        """
        Build SUMO lane IDs for junction at corridor position ``idx``.

        Uses the corridor naming convention from corridor.edg.xml:
          - Side streets: N{i}_to_J{i}, S{i}_to_J{i}
          - Arterial links: J{i}_to_J{i+1}, J{i+1}_to_J{i}
          - External entries: W1_to_J1, E1_to_J{N}

        Each edge has two lanes (_0, _1).
        """
        n = idx + 1  # 1-indexed junction number

        # North / South — always side streets
        north = [f"N{n}_to_{jid}_0", f"N{n}_to_{jid}_1"]
        south = [f"S{n}_to_{jid}_0", f"S{n}_to_{jid}_1"]

        # East approach
        if idx < self.n_junctions - 1:
            east_name = f"J{n+1}_to_{jid}"
        else:
            east_name = f"E1_to_{jid}"
        east = [f"{east_name}_0", f"{east_name}_1"]

        # West approach
        if idx > 0:
            west_name = f"J{n-1}_to_{jid}"
        else:
            west_name = f"W1_to_{jid}"
        west = [f"{west_name}_0", f"{west_name}_1"]

        return {"N": north, "S": south, "E": east, "W": west}

    # ── Web-Ready Decision Hooks ──────────────────────────────────────────

    def _log_decision(self, junction_id: str, phase: int, duration: int):
        """
        Log an RL decision for consumption by a future web layer.

        The web team can read ``self.decision_log`` and broadcast each
        entry over WebSocket to the admin panel.
        """
        phase_names = [
            "EW-Through", "EW-Left", "NS-Through", "NS-Left"
        ]

        event = {
            "type": "decision",
            "junction_id": junction_id,
            "phase": phase,
            "phase_name": phase_names[phase],
            "duration": duration,
            "sim_time": self.simulation_step,
            "timestamp": datetime.now().isoformat(),
            # Fields the web layer will add:
            # "override_deadline": timestamp + OVERRIDE_WINDOW_SECONDS
            # "override_applied": False
            # "final_duration": duration  (or overridden value)
        }
        self.decision_log.append(event)

    def get_latest_decisions(self, n: int = 10) -> List[Dict]:
        """
        Return the last N decision events (for web layer consumption).

        Args:
            n: Number of recent decisions to return.

        Returns:
            List of decision event dicts.
        """
        return self.decision_log[-n:]

    def get_current_state_snapshot(self) -> Dict:
        """
        Return a full state snapshot suitable for dashboard display.

        This is the data contract the web team should consume at 1 Hz.
        """
        states = self._get_all_traffic_states()

        snapshot = {
            "junctions": {},
            "system_status": "NORMAL",
            "sim_time": self.simulation_step,
            "cycle": self.cycle_count,
            "arrived": self.cumulative_arrived,
        }

        for jid in self.junction_ids:
            snapshot["junctions"][jid] = {
                "current_phase": self.current_phases[jid],
                "queues": {
                    d: states[jid]["queues"][i]
                    for i, d in enumerate(["N", "S", "E", "W"])
                },
                "wait_times": {
                    d: round(states[jid]["wait_times"][i], 1)
                    for i, d in enumerate(["N", "S", "E", "W"])
                },
                "emergency_flag": False,
            }

        return snapshot


# ── Quick self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("ILPS PER-PHASE CORRIDOR ENVIRONMENT — SELF TEST")
    print("=" * 70)

    env = PerPhaseCorridorEnv(n_junctions=3, use_gui=False)

    print(f"\n  Observation space: {env.observation_space}")
    print(f"  Action space:      {env.action_space}")
    print(f"  Junctions:         {env.junction_ids}")
    print(f"  Steps per cycle:   {env.n_junctions * env.n_phases} "
          f"({env.n_junctions} junctions × {env.n_phases} phases)")

    try:
        obs, info = env.reset()
        print(f"\n  Reset successful — obs shape: {obs.shape}")
        print(f"  Initial obs: {obs}")

        # Run one full cycle (3 junctions × 4 phases = 12 steps)
        total_reward = 0.0
        for step_i in range(env.n_junctions * env.n_phases):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            marker = " ← EXECUTED" if info["is_execution_step"] else ""
            print(f"  Step {step_i:2d}: J={info['junction_id']} "
                  f"P={info['phase']} D={info['duration']}s "
                  f"R={reward:+.2f}{marker}")

            if terminated:
                break

        print(f"\n  Cycle reward: {total_reward:.2f}")
        print(f"  Simulation time: {env.simulation_step}s")
        print(f"  Decision log: {len(env.decision_log)} events")

        env.close()
        print("\n  SUMO closed successfully")

    except Exception as e:
        print(f"\n  Test failed: {e}")
        import traceback
        traceback.print_exc()
        env.close()

    print("\n" + "=" * 70)
    print("  SELF TEST COMPLETE")
    print("=" * 70)
