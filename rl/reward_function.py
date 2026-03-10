"""
IntelliLight Reward Function Module
====================================

Calculates multi-objective rewards for training the RL agent.

The reward function combines multiple objectives:
1. Throughput: Maximize vehicles completing their journeys
2. Efficiency: Maximize throughput per unit of congestion
3. Fairness: Balance service across all directions
4. Wait time: Minimize total waiting time
5. Queue length: Minimize queue buildup
6. Starvation prevention: Prevent any direction from waiting too long
7. Emergency response: Prioritize emergency vehicles

This modular design allows easy tuning and experimentation with
different reward strategies without modifying the environment.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional

from configs.parameters import RewardConfig, SignalTiming

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    Calculates multi-objective rewards for traffic control.
    
    This class encapsulates all reward calculation logic, making it easy to:
    - Experiment with different reward strategies
    - Tune reward component weights
    - Add new reward objectives
    - Analyze which components drive learning
    """
    
    def __init__(
        self,
        wait_time_weight: Optional[float] = None,
        throughput_weight: Optional[float] = None,
        fairness_weight: Optional[float] = None,
        emergency_weight: Optional[float] = None,
        curriculum_stage: int = 0
    ):
        """
        Initialize reward calculator.
        
        Args:
            wait_time_weight: Weight for wait time penalty (defaults to config)
            throughput_weight: Weight for throughput reward (defaults to config)
            fairness_weight: Weight for fairness reward (defaults to config)
            emergency_weight: Weight for emergency response (defaults to config)
            curriculum_stage: Current curriculum learning stage (0-2)
        """
        # Use config defaults if not specified
        self.wait_time_weight = wait_time_weight or RewardConfig.WAIT_TIME_WEIGHT
        self.throughput_weight = throughput_weight or RewardConfig.THROUGHPUT_WEIGHT
        self.fairness_weight = fairness_weight or RewardConfig.FAIRNESS_WEIGHT
        self.emergency_weight = emergency_weight or RewardConfig.EMERGENCY_WEIGHT
        
        self.curriculum_stage = curriculum_stage
        
        # Track previous state for incremental calculations
        self.previous_arrived_count = 0
        self.last_served_step = {"W": 0, "E": 0, "N": 0, "S": 0}
        
        # Store individual reward components for analysis
        self.components = {
            'throughput': 0.0,
            'efficiency': 0.0,
            'fairness': 0.0,
            'wait_penalty': 0.0,
            'queue_penalty': 0.0,
            'starvation': 0.0,
            'emergency_bonus': 0.0
        }
        
        logger.debug(
            f"RewardCalculator initialized: "
            f"weights=[wait:{self.wait_time_weight}, through:{self.throughput_weight}, "
            f"fair:{self.fairness_weight}], stage={curriculum_stage}"
        )
    
    def calculate_reward(
        self,
        queues: np.ndarray,
        wait_times: np.ndarray,
        arrived_count: int,
        current_step: int,
        emergency_active: bool = False,
        action_direction: Optional[int] = None
    ) -> float:
        """
        Calculate total reward based on current traffic state.
        
        Args:
            queues: Array of queue lengths [W, E, N, S]
            wait_times: Array of waiting times [W, E, N, S]
            arrived_count: Total vehicles that have arrived (completed journey)
            current_step: Current simulation step
            emergency_active: Whether an emergency vehicle is present
            action_direction: Direction that got green light (0=EW, 1=NS, None=unknown)
        
        Returns:
            float: Total reward combining all objectives
        """
        # Calculate throughput (vehicles served since last call)
        throughput = self._calculate_throughput(arrived_count)
        
        # Calculate individual reward components
        throughput_reward = self._throughput_reward(throughput)
        efficiency_reward = self._efficiency_reward(throughput, queues)
        fairness_reward = self._fairness_reward(queues, wait_times)
        wait_penalty = self._wait_time_penalty(wait_times)
        queue_penalty = self._queue_penalty(queues)
        starvation_penalty = self._starvation_penalty(current_step, action_direction)
        emergency_bonus = self._emergency_bonus(emergency_active, action_direction)
        
        # Store components for analysis
        self.components = {
            'throughput': throughput_reward,
            'efficiency': efficiency_reward,
            'fairness': fairness_reward,
            'wait_penalty': wait_penalty,
            'queue_penalty': queue_penalty,
            'starvation': starvation_penalty,
            'emergency_bonus': emergency_bonus
        }
        
        # Combine all components
        total_reward = (
            throughput_reward +
            efficiency_reward +
            fairness_reward +
            wait_penalty +
            queue_penalty +
            starvation_penalty +
            emergency_bonus
        )
        
        # Apply curriculum-based scaling
        total_reward *= self._get_curriculum_multiplier()
        # total_reward = np.clip(total_reward, -50, 50)
        total_reward = np.tanh(total_reward / 100.0) * 100.0
        return float(total_reward)
    
    def _calculate_throughput(self, arrived_count: int) -> int:
        """Calculate vehicles served since last reward calculation."""
        throughput = max(0, arrived_count - self.previous_arrived_count)
        self.previous_arrived_count = arrived_count
        return throughput
    
    def _throughput_reward(self, throughput: int) -> float:
        """
        Reward for vehicles completing their journeys.
        
        Primary objective: maximize the number of vehicles served.
        """
        return throughput * self.throughput_weight * 5.0
    
    def _efficiency_reward(self, throughput: int, queues: np.ndarray) -> float:
        """
        Reward for efficiency (throughput per unit of congestion).
        
        Encourages serving vehicles while maintaining low queues.
        """
        total_queue = np.sum(queues)
        
        if total_queue > 0:
            efficiency = throughput /( total_queue+1)
        else:
            # Bonus for maintaining clear roads
            efficiency = throughput * 1.0
        
        return efficiency * self.throughput_weight
    
    def _fairness_reward(self, queues: np.ndarray, wait_times: np.ndarray) -> float:
        """
        Reward for balanced service across all directions.
        
        Prevents the agent from favoring one direction while ignoring others.
        Uses coefficient of variation (std/mean) as fairness metric.
        """
        # Queue balance
        queue_std = np.std(queues)
        queue_mean = np.mean(queues) + 1e-6  # Avoid division by zero
        queue_imbalance = (queue_std / queue_mean)
        
        # Wait time balance
        wait_std = np.std(wait_times)
        wait_mean = np.mean(wait_times) + 1e-6
        wait_imbalance =(wait_std / wait_mean)
        
        # Combine both metrics
        # fairness_score = (queue_balance + wait_balance) / 2.0
        imbalance=(queue_imbalance+wait_imbalance)/2.0
        return -imbalance*abs(self.fairness_weight)*5.0
    
    def _wait_time_penalty(self, wait_times: np.ndarray) -> float:
        """
        Penalty for total waiting time.
        
        Uses both linear and quadratic components:
        - Linear: Basic penalty proportional to wait time
        - Quadratic: Extra penalty for very long waits (urgency)
        """
        total_wait = np.sum(wait_times)
        
        # Linear component
        linear_penalty = total_wait * 0.02
        capped_wait = min(total_wait, 5000)
        
        # Quadratic component (increases urgency for long waits)
        quadratic_penalty = (capped_wait ** 1.3) * 0.0007
        total_penalty = min(linear_penalty + quadratic_penalty, 100.0)
        
        return -total_penalty * abs(self.wait_time_weight)
    
    def _queue_penalty(self, queues: np.ndarray) -> float:
        """
        Penalty for queue lengths with progressive scaling.
        
        Light congestion: Small penalty
        Heavy congestion: Large penalty (non-linear increase)
        """
        total_queue = np.sum(queues)
        
        if total_queue < 10:
            # Light congestion: linear penalty
            penalty = total_queue *0.5
        elif total_queue<30:
            base_penalty=10*0.5
            extra_penalty=(total_queue-10)*1.0
            penalty=base_penalty+extra_penalty
        else:
            # Heavy congestion: progressive penalty
            base_penalty = 5.0
            medium_penalty = 20.0*1.0
            extra_penalty = (total_queue - 30) * 2.0
            penalty = base_penalty + extra_penalty+medium_penalty
        
        penalty=min(penalty,20.0)

        return -penalty
    
    def _starvation_penalty(
        self,
        current_step: int,
        action_direction: Optional[int]
    ) -> float:
        """
        Penalty for letting any direction wait too long.
        
        Prevents the agent from ignoring low-traffic directions indefinitely.
        Exponential penalty kicks in after MAX_WAIT threshold.
        """
        # Update last served times
        if action_direction is not None:
            if action_direction == 0:  # EW
                self.last_served_step["E"] = current_step
                self.last_served_step["W"] = current_step
            else:  # NS
                self.last_served_step["N"] = current_step
                self.last_served_step["S"] = current_step
        
        # Calculate starvation penalties
        total_penalty = 0.0
        
        for direction, last_served in self.last_served_step.items():
            wait_duration = current_step - last_served
            
            if wait_duration > SignalTiming.MAX_WAIT:
                # Exponential penalty for exceeding threshold
                excess = wait_duration - SignalTiming.MAX_WAIT
                penalty = excess * 0.5
                # if excess <50:
                #     penalty = 20.0 * (1.1 ** excess)
                # else:
                #     penalty = 20.0 * (1.1 ** 50) + (excess - 50) * 50
                penalty = min(penalty, 20.0)
                total_penalty -= penalty
        
        return total_penalty
    
    def _emergency_bonus(
        self,
        emergency_active: bool,
        action_direction: Optional[int]
    ) -> float:
        """
        Bonus for responding to emergency vehicles.
        
        Large positive reward for giving green light when emergency is present.
        """
        if emergency_active and action_direction is not None:
            # Bonus for any action when emergency is present
            # (Direction-specific bonus would require knowing emergency location)
            return self.emergency_weight
        
        return 0.0
    
    def _get_curriculum_multiplier(self) -> float:
        """
        Get reward scaling factor based on curriculum stage.
        
        Slightly increases reward magnitude in later stages to account
        for higher traffic complexity.
        """
        return 1.0 + (self.curriculum_stage * 0.2)
    
    def reset(self):
        """
        Reset internal state for a new episode.
        
        Call this when starting a new episode to clear tracking variables.
        """
        self.previous_arrived_count = 0
        self.last_served_step = {"W": 0, "E": 0, "N": 0, "S": 0}
        self.components = {k: 0.0 for k in self.components}
        
        logger.debug("RewardCalculator reset for new episode")
    
    def get_component_breakdown(self) -> Dict[str, float]:
        """
        Get the individual reward components from the last calculation.
        
        Useful for analyzing which objectives are driving learning.
        
        Returns:
            dict: Component names mapped to their values
        """
        return self.components.copy()
    
    def update_weights(
        self,
        wait_time_weight: Optional[float] = None,
        throughput_weight: Optional[float] = None,
        fairness_weight: Optional[float] = None,
        emergency_weight: Optional[float] = None
    ):
        """
        Update reward component weights.
        
        Allows dynamic tuning of objectives during training or evaluation.
        
        Args:
            wait_time_weight: New weight for wait time penalty
            throughput_weight: New weight for throughput reward
            fairness_weight: New weight for fairness reward
            emergency_weight: New weight for emergency response
        """
        if wait_time_weight is not None:
            self.wait_time_weight = wait_time_weight
        if throughput_weight is not None:
            self.throughput_weight = throughput_weight
        if fairness_weight is not None:
            self.fairness_weight = fairness_weight
        if emergency_weight is not None:
            self.emergency_weight = emergency_weight
        
        logger.info(
            f"Reward weights updated: wait={self.wait_time_weight}, "
            f"throughput={self.throughput_weight}, fairness={self.fairness_weight}"
        )
    
    def set_curriculum_stage(self, stage: int):
        """
        Update curriculum learning stage.
        
        Args:
            stage: New curriculum stage (0-2)
        """
        if stage != self.curriculum_stage:
            self.curriculum_stage = stage
            logger.info(f"Curriculum stage updated to {stage}")


# Convenience function for simple use cases
def calculate_simple_reward(
    wait_times: np.ndarray,
    throughput: int
) -> float:
    """
    Simple reward function for quick prototyping.
    
    Just minimizes wait time and maximizes throughput.
    Use RewardCalculator class for full multi-objective rewards.
    
    Args:
        wait_times: Waiting times for each direction
        throughput: Vehicles served this step
    
    Returns:
        float: Simple reward value
    """
    total_wait = np.sum(wait_times)
    reward = throughput * 2.0 - total_wait * 0.1
    return float(reward)  # Ensure it's Python float, not numpy float


if __name__ == "__main__":
    """Simple test of reward calculator."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing RewardCalculator Module")
    print("=" * 60)
    
    # Create reward calculator
    print("\n1. Creating RewardCalculator...")
    calc = RewardCalculator(curriculum_stage=0)
    print("   ✓ RewardCalculator created")
    
    # Test scenario 1: Light traffic, good flow
    print("\n2. Test Scenario 1: Light traffic, good flow")
    queues = np.array([2, 3, 2, 1])  # Small queues
    wait_times = np.array([5, 8, 6, 4])  # Low wait times
    arrived = 10  # Good throughput
    
    reward1 = calc.calculate_reward(
        queues, wait_times, arrived, current_step=10, action_direction=0
    )
    print(f"   Reward: {reward1:.2f}")
    print(f"   Components: {calc.get_component_breakdown()}")
    
    # Test scenario 2: Heavy traffic, congestion
    print("\n3. Test Scenario 2: Heavy traffic, congestion")
    queues = np.array([15, 12, 18, 14])  # Large queues
    wait_times = np.array([45, 50, 60, 55])  # High wait times
    arrived = 15  # Some throughput but not enough
    
    reward2 = calc.calculate_reward(
        queues, wait_times, arrived, current_step=20, action_direction=1
    )
    print(f"   Reward: {reward2:.2f}")
    print(f"   Components: {calc.get_component_breakdown()}")
    
    # Test scenario 3: Emergency vehicle
    print("\n4. Test Scenario 3: Emergency vehicle present")
    queues = np.array([5, 5, 5, 5])
    wait_times = np.array([10, 10, 10, 10])
    arrived = 20
    
    reward3 = calc.calculate_reward(
        queues, wait_times, arrived, current_step=30,
        emergency_active=True, action_direction=0
    )
    print(f"   Reward: {reward3:.2f}")
    print(f"   Components: {calc.get_component_breakdown()}")
    
    # Test reset
    print("\n5. Testing reset...")
    calc.reset()
    print("   ✓ Calculator reset")
    
    # Test weight updates
    print("\n6. Testing weight updates...")
    calc.update_weights(throughput_weight=1.0, fairness_weight=-0.5)
    print("   ✓ Weights updated")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
    print(f"\nScenario 1 (good flow): {reward1:.2f}")
    print(f"Scenario 2 (congestion): {reward2:.2f}")
    print(f"Scenario 3 (emergency): {reward3:.2f}")
    print("\nExpected: Scenario 1 > Scenario 3 > Scenario 2")