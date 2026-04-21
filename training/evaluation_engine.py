"""
Evaluation Engine
=================

Runs evaluation episodes and collects performance metrics for
comparing different traffic controllers.

Supports:
- Running N episodes for any controller
- Collecting detailed metrics per episode
- Aggregating results across episodes
- Comparing RL vs baseline controllers
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from rl.multi_agent_env import CorridorEnv
from training.baseline_controllers import TrafficController
from training.metrics_calculator import MetricsCalculator, EpisodeMetrics

logger = logging.getLogger(__name__)


@dataclass
class AggregatedMetrics:
    """Metrics aggregated across multiple episodes."""
    
    mean: EpisodeMetrics
    std: Dict[str, float]
    min: Dict[str, float]
    max: Dict[str, float]
    n_episodes: int


class EvaluationEngine:
    """
    Runs evaluation episodes and collects metrics.
    
    This engine handles:
    - Running episodes with any controller
    - Collecting step-by-step data
    - Computing episode metrics
    - Aggregating across multiple episodes
    """
    
    def __init__(
        self,
        env: Optional[CorridorEnv] = None,
        metrics_calculator: Optional[MetricsCalculator] = None
    ):
        """
        Initialize evaluation engine.
        
        Args:
            env: Traffic environment (creates new if None)
            metrics_calculator: Metrics calculator (creates new if None)
        """
        self.env = env
        self.metrics_calc = metrics_calculator or MetricsCalculator()
        
        logger.info("EvaluationEngine initialized")
    
    def evaluate_controller(
        self,
        controller: TrafficController,
        n_episodes: int = 10,
        scenario: str = "RANDOM",
        curriculum_stage: int = 0,
        use_gui: bool = False,
        verbose: bool = True
    ) -> AggregatedMetrics:
        """
        Evaluate a controller over multiple episodes.
        
        Args:
            controller: Controller to evaluate
            n_episodes: Number of episodes to run
            scenario: Traffic scenario name
            curriculum_stage: Curriculum difficulty level
            use_gui: Show SUMO GUI (slow, for demo only)
            verbose: Print progress
        
        Returns:
            AggregatedMetrics: Results aggregated across episodes
        """
        logger.info(
            f"Evaluating {controller.get_name()} controller: "
            f"{n_episodes} episodes, scenario={scenario}"
        )
        
        # Create environment if needed
        if self.env is None:
            self.env = CorridorEnv(
                use_gui=use_gui,
                curriculum_stage=curriculum_stage
            )
        
        all_metrics = []
        
        for ep in range(n_episodes):
            if verbose:
                print(f"   Episode {ep+1}/{n_episodes}...", end=" ", flush=True)
            
            # Run episode
            episode_data = self._run_episode(
                controller=controller,
                scenario=scenario
            )
            
            # Calculate metrics
            metrics = self.metrics_calc.calculate_metrics(episode_data)
            all_metrics.append(metrics)
            
            if verbose:
                print(f" (throughput={metrics.throughput}, wait={metrics.avg_wait_time:.1f}s)")
        
        # Aggregate results
        aggregated = self._aggregate_metrics(all_metrics)
        
        logger.info(
            f"{controller.get_name()} evaluation complete: "
            f"avg_throughput={aggregated.mean.throughput:.1f}, "
            f"avg_wait={aggregated.mean.avg_wait_time:.1f}s"
        )
        
        return aggregated
    
    def _run_episode(
        self,
        controller: TrafficController,
        scenario: str
    ) -> Dict[str, List]:
        """
        Run single evaluation episode.
        
        Args:
            controller: Controller to use
            scenario: Traffic scenario
        
        Returns:
            dict: Episode data for metrics calculation
        """
        # Reset controller and environment
        controller.reset()
        obs, info = self.env.reset()
        
        # Data collection
        queues_history = []
        wait_times_history = []
        actions_history = []
        info_history = []
        phase_history = []
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get action from controller
            action = controller.select_action(obs, info)
            
            # Step environment
            obs, reward, done, truncated, info = self.env.step(action)
            phase = getattr(self.env, "current_phases", [0,0,0])
            if isinstance(phase, dict):
                phase_history.append(tuple(phase.values()))
            elif isinstance(phase, list):
                phase_history.append(tuple(phase))
            else:
                phase_history.append(phase)
            
            # Record data
            if 'intersections' in info:
                # Aggregate across all intersections
                all_queues = []
                all_wait_times = []
                for jid, j_info in info['intersections'].items():
                    all_queues.extend(j_info.get('queues', []))
                    all_wait_times.extend(j_info.get('wait_times', []))
                queues_history.append(all_queues)
                wait_times_history.append(all_wait_times)
                final_throughput = info.get('throughput', info.get('total_arrived', 0))
            else:
                queues_history.append(info.get('queues', []))
                wait_times_history.append(info.get('wait_times', []))
                final_throughput = info.get('throughput', info.get('arrived', 0))

            actions_history.append(action)
            info_history.append(info)
        
        episode_data = {
            'queues': queues_history,
            'wait_times': wait_times_history,
            'throughput': final_throughput,
            'actions': actions_history,
            'phases': phase_history,
            'info': info_history
        }
        
        return episode_data
    
    def _aggregate_metrics(
        self,
        metrics_list: List[EpisodeMetrics]
    ) -> AggregatedMetrics:
        """
        Aggregate metrics across multiple episodes.
        
        Computes mean, std, min, max for all metrics.
        
        Args:
            metrics_list: List of metrics from each episode
        
        Returns:
            AggregatedMetrics: Aggregated statistics
        """
        n_episodes = len(metrics_list)
        
        # Extract scalar values
        values = {
            'avg_wait_time': [m.avg_wait_time for m in metrics_list],
            'max_wait_time': [m.max_wait_time for m in metrics_list],
            'throughput': [m.throughput for m in metrics_list],
            'avg_queue_length': [m.avg_queue_length for m in metrics_list],
            'max_queue_length': [m.max_queue_length for m in metrics_list],
            'phase_switches': [m.phase_switches for m in metrics_list],
            'phase_switch_frequency': [m.phase_switch_frequency for m in metrics_list],
            'starvation_events': [m.starvation_events for m in metrics_list],
            'intersection_utilization': [m.intersection_utilization for m in metrics_list],
            'fairness_score': [m.fairness_score for m in metrics_list],
            'avg_cycle_time': [m.avg_cycle_time for m in metrics_list]
        }
        
        # Compute statistics
        mean_metrics = EpisodeMetrics()
        std_dict = {}
        min_dict = {}
        max_dict = {}
        
        for key, vals in values.items():
            setattr(mean_metrics, key, float(np.mean(vals)))
            std_dict[key] = float(np.std(vals))
            min_dict[key] = float(np.min(vals))
            max_dict[key] = float(np.max(vals))
        
        # Per-direction metrics (average across episodes)
        mean_metrics.wait_time_per_direction = self._average_dict_metrics(
            [m.wait_time_per_direction for m in metrics_list]
        )
        mean_metrics.max_wait_per_direction = self._average_dict_metrics(
            [m.max_wait_per_direction for m in metrics_list]
        )
        mean_metrics.queue_per_direction = self._average_dict_metrics(
            [m.queue_per_direction for m in metrics_list]
        )
        
        return AggregatedMetrics(
            mean=mean_metrics,
            std=std_dict,
            min=min_dict,
            max=max_dict,
            n_episodes=n_episodes
        )
    
    def _average_dict_metrics(
        self,
        dict_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Average dictionary values across episodes."""
        if not dict_list:
            return {}
        
        keys = dict_list[0].keys()
        result = {}
        
        for key in keys:
            values = [d[key] for d in dict_list if key in d]
            if values:
                result[key] = float(np.mean(values))
        
        return result
    
    def compare_controllers(
        self,
        rl_controller: TrafficController,
        baseline_controller: TrafficController,
        n_episodes: int = 10,
        scenarios: List[str] = None,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Compare RL controller vs baseline across scenarios.
        
        Args:
            rl_controller: RL-based controller
            baseline_controller: Baseline controller
            n_episodes: Episodes per scenario
            scenarios: List of scenarios to test
            verbose: Print progress
        
        Returns:
            dict: Comparison results per scenario
        """
        if scenarios is None:
            scenarios = ["MORNING_RUSH", "EVENING_RUSH", "WEEKEND"]
        
        results = {}
        
        for scenario in scenarios:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Scenario: {scenario}")
                print(f"{'='*60}")
            
            # Evaluate baseline
            if verbose:
                print(f"\n{baseline_controller.get_name()}:")
            baseline_metrics = self.evaluate_controller(
                baseline_controller,
                n_episodes=n_episodes,
                scenario=scenario,
                verbose=verbose
            )
            
            # Evaluate RL
            if verbose:
                print(f"\n{rl_controller.get_name()}:")
            rl_metrics = self.evaluate_controller(
                rl_controller,
                n_episodes=n_episodes,
                scenario=scenario,
                verbose=verbose
            )
            
            # Calculate improvements
            improvements = self.metrics_calc.compare_metrics(
                baseline_metrics.mean,
                rl_metrics.mean
            )
            
            results[scenario] = {
                'baseline': baseline_metrics,
                'rl': rl_metrics,
                'improvements': improvements
            }
            
            if verbose:
                print(f"\n Improvements:")
                for metric, improvement in improvements.items():
                    symbol = "" if improvement > 0 else ""
                    print(f"   {metric}: {improvement:+.1f}% {symbol}")
        
        return results
    
    def close(self):
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
            logger.debug("Evaluation environment closed")


if __name__ == "__main__":
    """Test evaluation engine."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing EvaluationEngine")
    print("=" * 60)
    
    from training.baseline_controllers import MaxPressureController, FixedTimerController
    
    # Create controllers
    mp = MaxPressureController(cyclic_mode=True)
    ft = FixedTimerController(cyclic_mode=True)
    
    # Create engine
    engine = EvaluationEngine()
    
    print("\n1. Evaluating Max-Pressure (3 episodes)...")
    mp_metrics = engine.evaluate_controller(
        mp,
        n_episodes=3,
        scenario="WEEKEND",
        verbose=True
    )
    
    print(f"\n   Results:")
    print(f"   Throughput: {mp_metrics.mean.throughput:.0f}  {mp_metrics.std['throughput']:.0f}")
    print(f"   Avg Wait: {mp_metrics.mean.avg_wait_time:.1f}s  {mp_metrics.std['avg_wait_time']:.1f}s")
    print(f"   Utilization: {mp_metrics.mean.intersection_utilization:.2%}")
    
    print("\n2. Evaluating Fixed-Timer (3 episodes)...")
    ft_metrics = engine.evaluate_controller(
        ft,
        n_episodes=3,
        scenario="WEEKEND",
        verbose=True
    )
    
    print(f"\n   Results:")
    print(f"   Throughput: {ft_metrics.mean.throughput:.0f}  {ft_metrics.std['throughput']:.0f}")
    print(f"   Avg Wait: {ft_metrics.mean.avg_wait_time:.1f}s  {ft_metrics.std['avg_wait_time']:.1f}s")
    print(f"   Utilization: {ft_metrics.mean.intersection_utilization:.2%}")
    
    # Compare
    calc = MetricsCalculator()
    improvements = calc.compare_metrics(ft_metrics.mean, mp_metrics.mean)
    
    print(f"\n3. Max-Pressure vs Fixed-Timer:")
    for metric, imp in improvements.items():
        print(f"   {metric}: {imp:+.1f}%")
    
    engine.close()
    
    print("\n" + "=" * 60)
    print(" Evaluation Engine working correctly!")
    print("=" * 60)