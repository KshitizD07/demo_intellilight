"""
Enhanced Reward Function for IntelliLight
==========================================

Improved production reward function with:

- Exponential anti-starvation penalty (controlled)
- Emergency vehicle priority
- Fairness and efficiency balancing
- Stable reward scaling for PPO
- Pressure-based reward component (stabilizes RL)

FIXES APPLIED:
- Corrected starvation penalty logic (removed double-negative)
- Reduced throughput multiplier (5.0 → 2.0)
- Simplified efficiency reward
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Reward component weights."""
    throughput: float = 1.0
    wait_time: float = -0.08
    queue: float = -0.04
    fairness: float = -0.15
    starvation: float = -8.0  # Negative (penalty)
    emergency: float = 100.0
    efficiency: float = 0.4
    pressure: float = 0.6


class EnhancedRewardCalculator:
    """
    Production-ready reward calculator.
    
    Key improvements over basic version:
    - Exponential starvation penalty (prevents 90+ second waits)
    - Emergency vehicle priority (100x reward)
    - Pressure-based component (faster convergence)
    - Safety violation tracking
    """
    
    def __init__(
        self,
        weights: RewardWeights = None,
        max_acceptable_wait: int = 90,
        emergency_max_wait: int = 30,
        enable_curriculum: bool = True
    ):
        """
        Initialize reward calculator.
        
        Args:
            weights: Reward component weights
            max_acceptable_wait: Maximum safe wait time (seconds)
            emergency_max_wait: Max acceptable emergency vehicle wait
            enable_curriculum: Use curriculum learning multiplier
        """
        self.weights = weights or RewardWeights()
        self.max_acceptable_wait = max_acceptable_wait
        self.emergency_max_wait = emergency_max_wait
        self.enable_curriculum = enable_curriculum
        
        # State tracking
        self.previous_arrived_count = 0
        self.curriculum_stage = 0
        
        # Safety metrics
        self.safety_violations = 0
        self.starvation_events = 0
    
    def calculate_reward(
        self,
        queues: List[int],
        wait_times: List[float],
        arrived_count: int,
        emergency_active: bool = False,
        emergency_direction: int = None,
        current_phase: int = 0
    ) -> float:
        """
        Calculate total reward for current state-action.
        
        Args:
            queues: Queue lengths [N, S, E, W]
            wait_times: Wait times [N, S, E, W] in seconds
            arrived_count: Cumulative vehicles that completed trip
            emergency_active: Is emergency vehicle present?
            emergency_direction: Which direction (0=N, 1=S, 2=E, 3=W)
            current_phase: Current phase (0=EW, 1=NS for 2-phase)
        
        Returns:
            Total reward (clipped to [-300, 120])
        """
        # Calculate throughput (vehicles served this step)
        throughput = self._calculate_throughput(arrived_count)
        
        # Reward components
        throughput_reward = self._throughput_reward(throughput)
        wait_penalty = self._wait_time_penalty(wait_times)
        queue_penalty = self._queue_penalty(queues)
        fairness_penalty = self._fairness_penalty(queues, wait_times)
        starvation_penalty = self._starvation_penalty(wait_times)
        emergency_reward = self._emergency_reward(
            emergency_active,
            emergency_direction,
            current_phase,
            wait_times
        )
        efficiency_reward = self._efficiency_reward(throughput, queues)
        pressure_reward = self._pressure_reward(queues)
        
        # Total reward
        total_reward = (
            throughput_reward +
            wait_penalty +
            queue_penalty +
            fairness_penalty +
            starvation_penalty +
            emergency_reward +
            efficiency_reward +
            pressure_reward
        )
        
        # Apply curriculum multiplier
        if self.enable_curriculum:
            total_reward *= self._get_curriculum_multiplier()
        
        # Clip to reasonable range for PPO stability
        total_reward = np.clip(total_reward, -300, 120)
        
        # Track safety violations
        if any(w > self.max_acceptable_wait for w in wait_times):
            self.safety_violations += 1
        
        return float(total_reward)
    
    def _calculate_throughput(self, arrived_count: int) -> int:
        """Calculate vehicles served since last step."""
        throughput = max(0, arrived_count - self.previous_arrived_count)
        self.previous_arrived_count = arrived_count
        return throughput
    
    def _throughput_reward(self, throughput: int) -> float:
        """Reward for vehicles served."""
        # FIXED: Reduced multiplier from 5.0 to 2.0
        return throughput * self.weights.throughput * 2.0
    
    def _wait_time_penalty(self, wait_times: List[float]) -> float:
        """Penalty for long wait times."""
        avg_wait = np.mean(wait_times)
        
        # Linear component
        linear = avg_wait * self.weights.wait_time
        
        # Quadratic component (penalizes very long waits more)
        quadratic = (avg_wait ** 2) / 120.0 * self.weights.wait_time
        
        penalty = linear + quadratic
        
        # Cap penalty to avoid reward explosion
        return max(penalty, -80.0)
    
    def _queue_penalty(self, queues: List[int]) -> float:
        """Penalty for long queues."""
        avg_queue = np.mean(queues)
        max_queue = max(queues)
        
        # Base penalty
        penalty = avg_queue * self.weights.queue
        
        # Extra penalty for very long queues
        if max_queue > 40:
            penalty += (max_queue - 40) * self.weights.queue * 3
        
        # Cap penalty
        return max(penalty, -25.0)
    
    def _fairness_penalty(self, queues, wait_times):
        """Penalty for imbalanced service across directions."""
        queue_std = np.std(queues)
        queue_mean = np.mean(queues) + 1  # Avoid division by zero
        
        wait_std = np.std(wait_times)
        wait_mean = np.mean(wait_times) + 1
        
        # Coefficient of variation
        imbalance = (queue_std / queue_mean + wait_std / wait_mean) / 2
        
        return imbalance * self.weights.fairness * 4.0
    
    def _starvation_penalty(self, wait_times):
        """
        EXPONENTIAL penalty for starvation (wait > 60s).
        
        FIXED: Removed double-negative bug.
        """
        total_penalty = 0
        
        for wait in wait_times:
            if wait > 60:
                excess = wait - 60
                
                # Exponential penalty: -8.0 * (excess^1.3)
                # Example: 102s wait → -8.0 * (42^1.3) = -952
                penalty = self.weights.starvation * (excess ** 1.3)
                total_penalty += penalty
                
                # Track starvation events
                if wait > self.max_acceptable_wait:
                    self.starvation_events += 1
        
        # FIXED: Don't use -abs(), weights.starvation is already negative
        return total_penalty
    
    def _emergency_reward(
        self,
        emergency_active,
        emergency_direction,
        current_phase,
        wait_times
    ):
        """Massive reward for prioritizing emergency vehicles."""
        if not emergency_active:
            return 0.0
        
        emergency_has_green = False
        
        # Check if emergency direction has green
        # Queue order: [N, S, E, W]
        # Phase 0 = EW (indices 2, 3)
        # Phase 1 = NS (indices 0, 1)
        
        if current_phase == 0 and emergency_direction in [2, 3]:  # EW green
            emergency_has_green = True
        elif current_phase == 1 and emergency_direction in [0, 1]:  # NS green
            emergency_has_green = True
        
        # Huge reward if emergency has green
        if emergency_has_green:
            return self.weights.emergency  # +100
        
        # Check emergency wait time
        emergency_wait = wait_times[emergency_direction]
        
        # Massive penalty if emergency waiting too long
        if emergency_wait > self.emergency_max_wait:
            return -self.weights.emergency * 2  # -200
        
        # Moderate penalty if emergency waiting
        return -self.weights.emergency * 0.5  # -50
    
    def _efficiency_reward(self, throughput, queues):
        """
        Reward for efficient operation (high throughput, low queues).
        
        FIXED: Simplified logic, removed inconsistent special case.
        """
        total_queue = sum(queues)
        
        # Efficiency = throughput / queues
        # Add 1 to avoid division by zero
        efficiency = throughput / (total_queue + 1)
        
        # Scale: typical efficiency is 0.5-2.0
        return efficiency * self.weights.efficiency * 5.0
    
    def _pressure_reward(self, queues):
        """
        Pressure-based component (inspired by Max-Pressure).
        
        Rewards balancing queues between EW and NS directions.
        Helps RL converge faster.
        """
        # Calculate pressure for each phase
        ew_pressure = queues[2] + queues[3]  # East + West
        ns_pressure = queues[0] + queues[1]  # North + South
        
        # Penalize imbalance
        pressure_balance = abs(ew_pressure - ns_pressure)
        
        return -pressure_balance * self.weights.pressure * 0.1
    
    def _get_curriculum_multiplier(self):
        """Curriculum learning multiplier."""
        return 1.0 + (self.curriculum_stage * 0.2)
    
    def set_curriculum_stage(self, stage: int):
        """Set curriculum stage (0, 1, or 2)."""
        self.curriculum_stage = max(0, min(2, stage))
    
    def reset(self):
        """Reset for new episode."""
        self.previous_arrived_count = 0
    
    def get_statistics(self) -> Dict:
        """Get safety statistics."""
        return {
            "safety_violations": self.safety_violations,
            "starvation_events": self.starvation_events,
            "curriculum_stage": self.curriculum_stage
        }


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED REWARD CALCULATOR TEST")
    print("=" * 70)
    
    calc = EnhancedRewardCalculator()
    
    # Test Case 1: Normal operation
    print("\n📊 TEST 1: Normal Operation")
    print("-" * 70)
    
    queues = [10, 12, 8, 10]  # Balanced
    wait_times = [25.0, 30.0, 20.0, 28.0]  # Reasonable
    arrived = 25
    
    reward = calc.calculate_reward(queues, wait_times, arrived)
    print(f"Queues: {queues}")
    print(f"Wait times: {wait_times}")
    print(f"Throughput: 25 vehicles")
    print(f"Total Reward: {reward:.2f}")
    
    # Test Case 2: Starvation scenario
    print("\n🚨 TEST 2: Starvation (102s wait)")
    print("-" * 70)
    
    calc.reset()
    queues = [5, 6, 4, 35]  # W direction starved
    wait_times = [15.0, 18.0, 12.0, 102.0]  # W waiting 102s!
    arrived = 20
    
    reward = calc.calculate_reward(queues, wait_times, arrived)
    print(f"Queues: {queues}")
    print(f"Wait times: {wait_times}")
    print(f"Throughput: 20 vehicles")
    print(f"Total Reward: {reward:.2f}")
    print(f"Starvation events: {calc.starvation_events}")
    
    # Test Case 3: Emergency vehicle
    print("\n🚑 TEST 3: Emergency Vehicle (has green)")
    print("-" * 70)
    
    calc.reset()
    queues = [10, 10, 8, 12]
    wait_times = [20.0, 22.0, 5.0, 18.0]
    arrived = 30
    
    reward = calc.calculate_reward(
        queues, wait_times, arrived,
        emergency_active=True,
        emergency_direction=2,  # East
        current_phase=0  # EW green
    )
    print(f"Queues: {queues}")
    print(f"Emergency direction: E (has green)")
    print(f"Total Reward: {reward:.2f} (+100 emergency bonus!)")
    
    # Test Case 4: Emergency vehicle waiting
    print("\n🚑 TEST 4: Emergency Vehicle (waiting)")
    print("-" * 70)
    
    calc.reset()
    queues = [10, 10, 8, 12]
    wait_times = [5.0, 8.0, 35.0, 18.0]  # Emergency waiting 35s
    arrived = 25
    
    reward = calc.calculate_reward(
        queues, wait_times, arrived,
        emergency_active=True,
        emergency_direction=2,  # East
        current_phase=1  # NS green (wrong phase!)
    )
    print(f"Queues: {queues}")
    print(f"Emergency direction: E (NO green, waiting 35s)")
    print(f"Total Reward: {reward:.2f} (-200 penalty!)")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)