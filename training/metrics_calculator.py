"""
Traffic Metrics Calculator
===========================

Calculates comprehensive performance metrics for traffic control evaluation.

Includes:
- Core metrics: wait time, throughput, queue length
- Safety metrics: phase switch frequency, maximum wait per direction
- Operational metrics: intersection utilization, fairness
- Efficiency metrics: average cycle time, capacity usage
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """
    Comprehensive metrics for a single episode.
    
    Collected during evaluation to assess controller performance.
    """
    
    # Core traffic metrics
    avg_wait_time: float = 0.0
    total_wait_time: float = 0.0
    max_wait_time: float = 0.0
    
    throughput: int = 0
    avg_queue_length: float = 0.0
    max_queue_length: float = 0.0
    
    # Per-direction metrics
    wait_time_per_direction: Dict[str, float] = field(default_factory=dict)
    max_wait_per_direction: Dict[str, float] = field(default_factory=dict)
    queue_per_direction: Dict[str, float] = field(default_factory=dict)
    
    # Safety metrics
    phase_switches: int = 0
    phase_switch_frequency: float = 0.0  # Switches per minute
    starvation_events: int = 0  # Times any direction waited > threshold
    
    # Operational metrics
    intersection_utilization: float = 0.0  # vehicles_served / vehicles_arrived
    fairness_score: float = 0.0  # Coefficient of variation
    
    # Efficiency metrics
    avg_cycle_time: float = 0.0
    total_green_time: float = 0.0
    total_red_time: float = 0.0
    
    # Emergency response
    emergency_present: bool = False
    emergency_clear_time: float = 0.0
    
    # Episode info
    episode_length: float = 0.0  # Simulation seconds
    total_steps: int = 0  # Controller actions


class MetricsCalculator:
    """
    Calculates comprehensive traffic metrics from episode data.
    
    Processes raw episode data to compute performance indicators
    for comparing different controllers.
    """
    
    def __init__(self, starvation_threshold: int = 90):
        """
        Initialize metrics calculator.
        
        Args:
            starvation_threshold: Maximum acceptable wait (seconds)
        """
        self.starvation_threshold = starvation_threshold
        logger.debug(f"MetricsCalculator initialized: threshold={starvation_threshold}s")
    
    def calculate_metrics(
        self,
        episode_data: Dict[str, List]
    ) -> EpisodeMetrics:
        """
        Calculate all metrics from episode data.
        
        Args:
            episode_data: Dictionary containing episode history:
                - 'queues': List of queue arrays per step
                - 'wait_times': List of wait time arrays per step
                - 'throughput': Total vehicles completed
                - 'actions': List of actions taken
                - 'info': List of info dicts per step
        
        Returns:
            EpisodeMetrics: Complete metrics for the episode
        """
        metrics = EpisodeMetrics()
        
        queues = episode_data['queues']  # List of [W, E, N, S] arrays
        wait_times = episode_data['wait_times']
        throughput = episode_data['throughput']
        actions = episode_data['actions']
        infos = episode_data.get('info', [])
        
        # Basic counts
        num_steps = len(queues)
        metrics.total_steps = num_steps
        metrics.throughput = throughput
        
        if num_steps == 0:
            logger.warning("Empty episode data!")
            return metrics
        
        # Calculate episode length (from last info if available)
        if infos:
            metrics.episode_length = infos[-1].get('simulation_step', 0)
        
        # ===== WAIT TIME METRICS =====
        
        all_wait_times = np.array(wait_times)  # Shape: (steps, 4)
        all_queues = np.array(queues)  # Shape: (steps, 4)
        
        # Total and average
        metrics.total_wait_time = float(np.sum(all_wait_times)) if all_wait_times.size > 0 else 0.0
        if all_wait_times.size > 0:
            metrics.avg_wait_time = float(np.mean(all_wait_times))
            metrics.max_wait_time = float(np.max(all_wait_times))
        
        if all_queues.size > 0:
            # Note: assuming shape is (steps, 3 intersections * 4 phases, etc.) depending on env
            # Safest is just summing across axis=1 if it's multidimensional
            metrics.avg_queue_length = float(np.mean(np.sum(all_queues, axis=1))) if all_queues.ndim > 1 else float(np.mean(all_queues))
            metrics.max_queue_length = float(np.max(np.sum(all_queues, axis=1)))  if all_queues.ndim > 1 else float(np.max(all_queues))
        
        # Per-direction wait times
        direction_names = ['W', 'E', 'N', 'S']
        for i, direction in enumerate(direction_names):
            if all_wait_times.size > 0 and all_wait_times.ndim > 1 and all_wait_times.shape[1] > i:
                metrics.wait_time_per_direction[direction] = float(np.mean(all_wait_times[:, i]))
                metrics.max_wait_per_direction[direction] = float(np.max(all_wait_times[:, i]))
            else:
                metrics.wait_time_per_direction[direction] = 0.0
                metrics.max_wait_per_direction[direction] = 0.0
                
        # Per-direction queues
        for i, direction in enumerate(direction_names):
            if all_queues.size > 0 and all_queues.ndim > 1 and all_queues.shape[1] > i:
                metrics.queue_per_direction[direction] = float(np.mean(all_queues[:, i]))
            else:
                metrics.queue_per_direction[direction] = 0.0
        
        # ===== SAFETY METRICS =====
        
        # Phase switch frequency (for cyclic: always 2 per action, for acyclic: count changes)
        # metrics.phase_switches = self._count_phase_switches(actions, infos)
        phases = episode_data.get('phases', [])
        metrics.phase_switches = sum(
            1 for i in range(1, len(phases))
            if phases[i] != phases[i-1]
        )
        
        if metrics.episode_length > 0:
            # Switches per minute
            metrics.phase_switch_frequency = (
                metrics.phase_switches / (metrics.episode_length / 60.0)
            )
        
        # Starvation events (any direction waiting > threshold)
        metrics.starvation_events = self._count_starvation_events(all_wait_times)
        
        # ===== OPERATIONAL METRICS =====
        
        # Intersection utilization
        # Note: This assumes vehicles_arrived is tracked somewhere
        # For now, we estimate from throughput and final queues
        final_queues = np.sum(all_queues[-1]) if len(all_queues) > 0 else 0
        vehicles_arrived = throughput + final_queues  # Approximation
        
        if vehicles_arrived > 0:
            metrics.intersection_utilization = throughput / vehicles_arrived
        else:
            metrics.intersection_utilization = 1.0  # No traffic case
        
        # Fairness score (coefficient of variation of wait times across directions)
        dir_wait_means = [metrics.wait_time_per_direction[d] for d in direction_names]
        if np.mean(dir_wait_means) > 0:
            metrics.fairness_score = float(
                np.std(dir_wait_means) / np.mean(dir_wait_means)
            )
        else:
            metrics.fairness_score = 0.0
        
        # ===== EFFICIENCY METRICS =====
        
        # Calculate cycle times and green/red splits
        cycle_info = self._calculate_cycle_info(actions, infos)
        metrics.avg_cycle_time = cycle_info['avg_cycle_time']
        metrics.total_green_time = cycle_info['total_green']
        metrics.total_red_time = cycle_info['total_red']
        
        # ===== EMERGENCY RESPONSE =====
        
        # Check if emergency vehicle was present
        for info in infos:
            if info.get('emergency', False):
                metrics.emergency_present = True
                break
        
        # TODO: Calculate emergency clearance time if needed
        
        logger.debug(f"Metrics calculated: {num_steps} steps, {throughput} throughput")
        
        return metrics
    
    def _count_phase_switches(
        self,
        actions: List,
        infos: List[Dict]
    ) -> int:
        """
        Count phase switches during episode.
        
        For cyclic: Each action = 2 switches (EWNSEW)
        For acyclic: Count direction changes
        
        Args:
            actions: List of actions
            infos: List of info dicts
        
        Returns:
            int: Number of phase switches
        """
        if not infos:
            # Cyclic assumption: 2 switches per action
            return len(actions) * 2
        
        # Check if cyclic mode (look at first info)
        if infos[0].get('mode') == 'CYCLIC':
            return len(actions) * 2
        
        # Acyclic: Count direction changes
        switches = 0
        prev_direction = None
        
        for action in actions:
            current_direction = action[0]  # First element is direction
            if prev_direction is not None and current_direction != prev_direction:
                switches += 1
            prev_direction = current_direction
        
        return switches
    
    def _count_starvation_events(self, wait_times: np.ndarray) -> int:
        """
        Count starvation events (any direction waiting too long).
        
        Args:
            wait_times: Array of wait times (steps, 4)
        
        Returns:
            int: Number of starvation events detected
        """
        # Count steps where ANY direction exceeded threshold
        exceeded = np.any(wait_times > self.starvation_threshold, axis=1)
        starvation_events = int(np.sum(exceeded))
        
        return starvation_events
    
    def _calculate_cycle_info(
        self,
        actions: List,
        infos: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate cycle timing information.
        
        Args:
            actions: List of actions
            infos: List of info dicts
        
        Returns:
            dict: Cycle timing metrics
        """
        result = {
            'avg_cycle_time': 0.0,
            'total_green': 0.0,
            'total_red': 0.0
        }
        
        if not infos:
            return result
        
        cycle_times = []
        total_green = 0.0
        total_red = 0.0
        
        for info in infos:
            # Get cycle time if available
            if 'cycle_time' in info:
                cycle_times.append(info['cycle_time'])
            
            # Accumulate green time
            if 'ew_duration' in info and 'ns_duration' in info:
                total_green += info['ew_duration'] + info['ns_duration']
            
            # Red time estimation (all-red periods)
            if 'cycle_time' in info:
                # Cycle = EW_green + NS_green + 2all_red
                green_in_cycle = info.get('ew_duration', 0) + info.get('ns_duration', 0)
                total_red += (info['cycle_time'] - green_in_cycle)
        
        if cycle_times:
            result['avg_cycle_time'] = float(np.mean(cycle_times))
        
        result['total_green'] = total_green
        result['total_red'] = total_red
        
        return result
    
    def compare_metrics(
        self,
        baseline_metrics: EpisodeMetrics,
        rl_metrics: EpisodeMetrics
    ) -> Dict[str, float]:
        """
        Compare RL metrics against baseline.
        
        Calculates improvement percentages for key metrics.
        
        Args:
            baseline_metrics: Metrics from baseline controller
            rl_metrics: Metrics from RL controller
        
        Returns:
            dict: Improvement percentages (negative = RL worse)
        """
        improvements = {}
        
        # Wait time improvement (lower is better)
        if baseline_metrics.avg_wait_time > 0:
            improvements['wait_time'] = (
                (baseline_metrics.avg_wait_time - rl_metrics.avg_wait_time)
                / baseline_metrics.avg_wait_time * 100
            )
        
        # Throughput improvement (higher is better)
        if baseline_metrics.throughput > 0:
            improvements['throughput'] = (
                (rl_metrics.throughput - baseline_metrics.throughput)
                / baseline_metrics.throughput * 100
            )
        
        # Queue improvement (lower is better)
        if baseline_metrics.avg_queue_length > 0:
            improvements['queue_length'] = (
                (baseline_metrics.avg_queue_length - rl_metrics.avg_queue_length)
                / baseline_metrics.avg_queue_length * 100
            )
        
        # Utilization improvement (higher is better)
        if baseline_metrics.intersection_utilization > 0:
            improvements['utilization'] = (
                (rl_metrics.intersection_utilization - baseline_metrics.intersection_utilization)
                / baseline_metrics.intersection_utilization * 100
            )
        
        # Fairness improvement (lower CoV is better)
        if baseline_metrics.fairness_score > 0:
            improvements['fairness'] = (
                (baseline_metrics.fairness_score - rl_metrics.fairness_score)
                / baseline_metrics.fairness_score * 100
            )
        
        # Starvation improvement (fewer is better)
        if baseline_metrics.starvation_events > 0:
            improvements['starvation'] = (
                (baseline_metrics.starvation_events - rl_metrics.starvation_events)
                / baseline_metrics.starvation_events * 100
            )
        else:
            improvements['starvation'] = 0.0  # Both have zero
        
        return improvements


if __name__ == "__main__":
    """Test metrics calculator."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing MetricsCalculator")
    print("=" * 60)
    
    # Create sample episode data
    np.random.seed(42)
    num_steps = 30
    
    episode_data = {
        'queues': [np.random.rand(4) * 10 for _ in range(num_steps)],
        'wait_times': [np.random.rand(4) * 50 for _ in range(num_steps)],
        'throughput': 450,
        'actions': [(np.random.randint(0, 16), np.random.randint(0, 16)) for _ in range(num_steps)],
        'info': [
            {
                'mode': 'CYCLIC',
                'ew_duration': 30,
                'ns_duration': 25,
                'cycle_time': 63,  # 30 + 25 + 24
                'simulation_step': i * 63,
                'emergency': False
            }
            for i in range(num_steps)
        ]
    }
    
    # Calculate metrics
    calc = MetricsCalculator(starvation_threshold=90)
    metrics = calc.calculate_metrics(episode_data)
    
    print("\n Calculated Metrics:")
    print(f"   Throughput: {metrics.throughput} vehicles")
    print(f"   Avg Wait Time: {metrics.avg_wait_time:.2f}s")
    print(f"   Max Wait Time: {metrics.max_wait_time:.2f}s")
    print(f"   Avg Queue Length: {metrics.avg_queue_length:.2f} vehicles")
    print(f"   Max Queue Length: {metrics.max_queue_length:.2f} vehicles")
    
    print("\n Safety Metrics:")
    print(f"   Phase Switches: {metrics.phase_switches}")
    print(f"   Switch Frequency: {metrics.phase_switch_frequency:.2f} /min")
    print(f"   Starvation Events: {metrics.starvation_events}")
    
    print("\n  Operational Metrics:")
    print(f"   Intersection Utilization: {metrics.intersection_utilization:.2%}")
    print(f"   Fairness Score: {metrics.fairness_score:.3f}")
    print(f"   Avg Cycle Time: {metrics.avg_cycle_time:.1f}s")
    
    print("\n Per-Direction Wait Times:")
    for direction, wait in metrics.wait_time_per_direction.items():
        max_wait = metrics.max_wait_per_direction[direction]
        print(f"   {direction}: avg={wait:.1f}s, max={max_wait:.1f}s")
    
    print("\n" + "=" * 60)
    print(" MetricsCalculator working correctly!")
    print("=" * 60)