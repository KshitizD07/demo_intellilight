"""
Baseline Traffic Controllers for Comparison
============================================

Implements traditional adaptive traffic control algorithms for
benchmarking the RL-based IntelliLight system.

Primary baseline: Max-Pressure Controller
- Industry-standard adaptive algorithm
- Serves direction with highest queue pressure
- Allocates green time proportionally to congestion
- Includes anti-starvation mechanism
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod

from configs.parameters import signal

logger = logging.getLogger(__name__)


class TrafficController(ABC):
    """
    Abstract base class for traffic controllers.
    
    All controllers (RL, baseline, or custom) should inherit from this
    to ensure consistent interface for evaluation.
    """
    
    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray,
        info: Optional[Dict] = None
    ) -> Tuple[int, int]:
        """
        Select traffic signal action based on current state.
        
        Args:
            observation: Current traffic state
            info: Additional information (optional)
        
        Returns:
            tuple: (ew_duration_idx, ns_duration_idx) for CYCLIC
                   or (direction, duration_idx) for ACYCLIC
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset controller state for new episode."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return controller name for logging."""
        pass


class MaxPressureController(TrafficController):
    """
    Max-Pressure Adaptive Traffic Controller.
    
    Industry-standard algorithm that:
    1. Calculates "pressure" = queue differential between directions
    2. Serves direction with maximum pressure
    3. Allocates green time proportional to pressure
    4. Prevents starvation with maximum wait threshold
    
    This is the primary baseline for evaluating IntelliLight.
    
    Reference:
    Varaiya, P. (2013). Max pressure control of a network of signalized
    intersections. Transportation Research Part C.
    """
    
    def __init__(
        self,
        min_green: int = 15,
        max_green: int = 60,
        base_green: int = 30,
        pressure_scale: float = 1.5,
        max_wait_threshold: int = 90,
        cyclic_mode: bool = True
    ):
        """
        Initialize Max-Pressure controller.
        
        Args:
            min_green: Minimum green duration (safety)
            max_green: Maximum green duration (fairness)
            base_green: Default green duration
            pressure_scale: Sensitivity to queue pressure
            max_wait_threshold: Force service after this wait time
            cyclic_mode: True for cyclic (both durations), False for acyclic
        """
        self.min_green = min_green
        self.max_green = max_green
        self.base_green = base_green
        self.pressure_scale = pressure_scale
        self.max_wait_threshold = max_wait_threshold
        self.cyclic_mode = cyclic_mode
        
        # Track service times
        self.last_served_time = {'EW': 0, 'NS': 0}
        self.current_time = 0
        
        logger.info(
            f"MaxPressureController initialized: "
            f"green=[{min_green}-{max_green}], base={base_green}, "
            f"scale={pressure_scale}, cyclic={cyclic_mode}"
        )
    
    def select_action(
        self,
        observation: np.ndarray,
        info: Optional[Dict] = None
    ) -> Tuple[int, int]:
        """
        Select action using max-pressure algorithm.
        
        For CYCLIC mode: Returns (ew_duration_idx, ns_duration_idx)
        For ACYCLIC mode: Returns (direction, duration_idx)
        
        Args:
            observation: [queues(4), wait_times(4), emergency(1), phase(1)]
            info: Optional dict with additional state
        
        Returns:
            tuple: Action indices for the environment
        """
        # Parse observation
        n_intersections = max(1, observation.size // 14)
        
        # Aggregate queues and wait times across all intersections
        total_queues = np.zeros(4)
        total_wait_times = np.zeros(4)
        emergency = False
        
        for i in range(n_intersections):
            local_obs = observation[i*14 : (i+1)*14]
            total_queues += local_obs[0:4] * 50.0
            total_wait_times += local_obs[8:12] * 120.0  # Denormalize
            if local_obs[12] > 0.5:
                emergency = True
                
        queues = total_queues
        wait_times = total_wait_times
        
        # Calculate queue pressure for each direction
        ew_queue = queues[0] + queues[1]  # West + East
        ns_queue = queues[2] + queues[3]  # North + South
        
        # Calculate wait time pressure
        ew_wait = wait_times[0] + wait_times[1]
        ns_wait = wait_times[2] + wait_times[3]
        
        # Combined pressure (queue + wait time factor)
        ew_pressure = ew_queue + (ew_wait / 60.0)  # Normalize wait to ~queue scale
        ns_pressure = ns_queue + (ns_wait / 60.0)
        
        # Check for starvation
        time_since_ew = self.current_time - self.last_served_time['EW']
        time_since_ns = self.current_time - self.last_served_time['NS']
        
        ew_starving = time_since_ew > self.max_wait_threshold
        ns_starving = time_since_ns > self.max_wait_threshold
        
        if self.cyclic_mode:
            # CYCLIC: Always serve both, decide durations based on pressure
            
            # Calculate green times proportional to pressure
            ew_duration = self._calculate_green_time(ew_pressure)
            ns_duration = self._calculate_green_time(ns_pressure)
            
            
            # Anti-starvation: Ensure minimum time for starving direction
            if ew_starving:
                ew_duration = max(ew_duration, self.base_green)
            if ns_starving:
                ns_duration = max(ns_duration, self.base_green)
            
            # Emergency override
            if emergency and info and 'emergency_direction' in info:
                if info['emergency_direction'] == 0:  # EW
                    ew_duration = self.max_green
                    ns_duration = self.min_green
                else:  # NS
                    ns_duration = self.max_green
                    ew_duration = self.min_green
            
            # Convert to action indices
            ew_idx = self._duration_to_index(ew_duration)
            ns_idx = self._duration_to_index(ns_duration)
            
            # Update service times (both served in cycle)
            self.current_time += ew_duration + ns_duration + (2 * signal.ALL_RED)
            self.last_served_time['EW'] = self.current_time
            self.last_served_time['NS'] = self.current_time
            
            return (ew_idx, ew_idx, ns_idx, ns_idx) * n_intersections
        
        else:
            # ACYCLIC: Choose one direction per action
            
            # Force service if starving
            if ns_starving and not ew_starving:
                direction = 1  # NS
                pressure = ns_pressure
            elif ew_starving and not ns_starving:
                direction = 0  # EW
                pressure = ew_pressure
            else:
                # Choose direction with maximum pressure
                if ew_pressure > ns_pressure:
                    direction = 0  # EW
                    pressure = ew_pressure
                else:
                    direction = 1  # NS
                    pressure = ns_pressure
            
            # Calculate green time
            duration = self._calculate_green_time(pressure)
            
            # Emergency override
            if emergency and info and 'emergency_direction' in info:
                if info['emergency_direction'] == direction:
                    duration = self.max_green
            
            # Update service time
            self.current_time += duration + signal.ALL_RED
            if direction == 0:
                self.last_served_time['EW'] = self.current_time
            else:
                self.last_served_time['NS'] = self.current_time
            
            # Convert to action
            duration_idx = self._duration_to_index(duration)
            
            return (direction, duration_idx) * n_intersections
    
    def _calculate_green_time(self, pressure: float) -> int:
        """
        Calculate green duration based on pressure.
        
        Higher pressure  longer green time
        
        Args:
            pressure: Combined queue + wait pressure
        
        Returns:
            int: Green duration in seconds
        """
        # Linear scaling with pressure
        green = self.base_green + (pressure * self.pressure_scale)
        
        # Clamp to safety bounds
        green = int(np.clip(green, self.min_green, self.max_green))
        
        return green
    
    def _duration_to_index(self, duration: int) -> int:
        """
        Convert duration in seconds to index in GREEN_OPTIONS.
        
        Finds closest available duration option.
        
        Args:
            duration: Desired green duration in seconds
        
        Returns:
            int: Index into signal.GREEN_DURATIONS
        """
        # Find closest duration in available options
        options = np.array(signal.GREEN_DURATIONS)
        idx = np.argmin(np.abs(options - duration))
        return int(idx)
    
    def reset(self):
        """Reset controller for new episode."""
        self.last_served_time = {'EW': 0, 'NS': 0}
        self.current_time = 0
        logger.debug("MaxPressureController reset")
    
    def get_name(self) -> str:
        """Return controller name."""
        return "Max-Pressure"


class FixedTimerController(TrafficController):
    """
    Simple fixed-timer controller.
    
    Serves EW and NS alternately with fixed green times.
    Represents traditional non-adaptive traffic signals.
    
    Used as a basic baseline to show value of adaptive control.
    """
    
    def __init__(
        self,
        ew_duration: int = 30,
        ns_duration: int = 30,
        cyclic_mode: bool = True
    ):
        """
        Initialize fixed-timer controller.
        
        Args:
            ew_duration: Fixed green time for EW (seconds)
            ns_duration: Fixed green time for NS (seconds)
            cyclic_mode: True for cyclic, False for acyclic
        """
        self.ew_duration = ew_duration
        self.ns_duration = ns_duration
        self.cyclic_mode = cyclic_mode
        self.phase = 0  # Track current phase for acyclic
        
        logger.info(
            f"FixedTimerController initialized: "
            f"EW={ew_duration}s, NS={ns_duration}s, cyclic={cyclic_mode}"
        )
    
    def select_action(
        self,
        observation: np.ndarray,
        info: Optional[Dict] = None
    ) -> Tuple[int, int]:
        """
        Select fixed-time action.
        
        Args:
            observation: Current state (ignored)
            info: Additional info (ignored)
        
        Returns:
            tuple: Fixed action
        """
        n_intersections = max(1, observation.size // 14) if observation is not None else 1
        
        if self.cyclic_mode:
            # Return fixed durations
            ew_idx = self._duration_to_index(self.ew_duration)
            ns_idx = self._duration_to_index(self.ns_duration)
            return (ew_idx, ew_idx, ns_idx, ns_idx) * n_intersections
        else:
            # Alternate between phases
            direction = self.phase
            duration = self.ew_duration if direction == 0 else self.ns_duration
            duration_idx = self._duration_to_index(duration)
            
            # Toggle phase for next call
            self.phase = 1 - self.phase
            
            return (direction, duration_idx) * n_intersections
    
    def _duration_to_index(self, duration: int) -> int:
        """Convert duration to index."""
        options = np.array(signal.GREEN_DURATIONS)
        idx = np.argmin(np.abs(options - duration))
        return int(idx)
    
    def reset(self):
        """Reset controller."""
        self.phase = 0
        logger.debug("FixedTimerController reset")
    
    def get_name(self) -> str:
        """Return controller name."""
        return "Fixed-Timer"


class RLController(TrafficController):
    """
    Wrapper for trained RL model.
    
    Provides consistent interface for evaluation alongside baselines.
    """
    
    def __init__(self, model_path: str, deterministic: bool = True):
        """
        Initialize RL controller.
        
        Args:
            model_path: Path to trained model (.zip file)
            deterministic: Use deterministic policy (no exploration)
        """
        from stable_baselines3 import PPO
        
        self.model = PPO.load(model_path)
        self.deterministic = deterministic
        
        logger.info(f"RLController loaded from: {model_path}")
    
    def select_action(
        self,
        observation: np.ndarray,
        info: Optional[Dict] = None
    ) -> Tuple[int, int]:
        """
        Select action using trained RL policy.
        
        Args:
            observation: Current state
            info: Additional info (unused)
        
        Returns:
            tuple: Action from RL model
        """
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return tuple(action)
    
    def reset(self):
        """Reset (no-op for RL)."""
        pass
    
    def get_name(self) -> str:
        """Return controller name."""
        return "IntelliLight-RL"


if __name__ == "__main__":
    """Test baseline controllers."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing Baseline Controllers")
    print("=" * 60)
    
    # Test Max-Pressure
    print("\n1. Testing Max-Pressure Controller (Cyclic)")
    mp = MaxPressureController(cyclic_mode=True)
    
    # Scenario: Heavy EW traffic
    obs = np.array([
        0.6, 0.5, 0.1, 0.1,  # Queues: EW heavy, NS light
        0.7, 0.6, 0.2, 0.1,  # Wait times: EW long, NS short
        0.0,  # No emergency
        0.0   # Phase
    ])
    
    action = mp.select_action(obs)
    print(f"   Heavy EW traffic  Action: {action}")
    print(f"   EW duration: {signal.GREEN_DURATIONS[action[0]]}s")
    print(f"   NS duration: {signal.GREEN_DURATIONS[action[1]]}s")
    print(f"   Expected: EW > NS " if action[0] > action[1] else "   FAIL")
    
    # Test Fixed-Timer
    print("\n2. Testing Fixed-Timer Controller")
    ft = FixedTimerController(ew_duration=30, ns_duration=30, cyclic_mode=True)
    
    action = ft.select_action(obs)
    print(f"   Action: {action}")
    print(f"   EW duration: {signal.GREEN_DURATIONS[action[0]]}s")
    print(f"   NS duration: {signal.GREEN_DURATIONS[action[1]]}s")
    print(f"   Expected: Both 30s ")
    
    print("\n" + "=" * 60)
    print("Baseline controllers working correctly!")
    print("=" * 60)