"""
IntelliLight Model Evaluation Script
=====================================

Comprehensive evaluation of trained IntelliLight model vs baselines.

Usage:
    python training/evaluate_model.py --model models/checkpoints/intellilight_final.zip --episodes 20

Features:
- Evaluates RL agent vs Max-Pressure and Fixed-Timer baselines
- Tests across multiple traffic scenarios
- Generates detailed performance report
- Saves results to JSON for analysis
"""

import argparse
import json
import logging
import sys

from pathlib import Path
from datetime import datetime
from typing import Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from rl.traffic_env import TrafficEnv4Phase
from training.baseline_controllers import (
    MaxPressureController,
    FixedTimerController,
    RLController
)
from training.evaluation_engine import EvaluationEngine
from training.metrics_calculator import MetricsCalculator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    n_episodes: int = 20,
    scenarios: list = None,
    output_file: str = None,
    use_gui: bool = False
) -> Dict:
    """
    Run comprehensive evaluation of trained model.
    
    Args:
        model_path: Path to trained model .zip file
        n_episodes: Episodes per scenario per controller
        scenarios: List of scenarios to test
        output_file: Path to save JSON results (optional)
        use_gui: Show SUMO GUI (slow, for demo only)
    
    Returns:
        dict: Complete evaluation results
    """
    if scenarios is None:
        scenarios = ["MORNING_RUSH", "EVENING_RUSH", "WEEKEND"]
    
    print("=" * 70)
    print("INTELLILIGHT MODEL EVALUATION")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Episodes per scenario: {n_episodes}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Evaluation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Create controllers
    print("\n Loading controllers...")
    
    rl_controller = RLController(model_path, deterministic=True)
    print(f"    RL Controller loaded")
    
    max_pressure = MaxPressureController(cyclic_mode=True)
    print(f"    Max-Pressure Controller created")
    
    fixed_timer = FixedTimerController(ew_duration=30, ns_duration=30, cyclic_mode=True)
    print(f"    Fixed-Timer Controller created")
    
    # Create evaluation engine
    print("\n Initializing evaluation engine...")
    engine = EvaluationEngine()
    print("    Engine ready")
    
    # Store all results
    all_results = {
        'model_path': model_path,
        'n_episodes': n_episodes,
        'scenarios': scenarios,
        'timestamp': datetime.now().isoformat(),
        'results_by_scenario': {}
    }
    
    # Evaluate each scenario
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f" SCENARIO: {scenario}")
        print(f"{'='*70}")
        
        scenario_results = {}
        
        scenario_metrics = {}
        # Evaluate all Controllers
        for controller in [fixed_timer, max_pressure, rl_controller]:
            print(f"\n{controller.get_name()}:")
            
            metrics = engine.evaluate_controller(
                controller,
                n_episodes=n_episodes,
                scenario=scenario,
                use_gui=use_gui,
                verbose=True
            )
            
            scenario_metrics[controller.get_name()] = metrics
            
            scenario_results[controller.get_name()] = {
                'mean': {
                    'avg_wait_time': metrics.mean.avg_wait_time,
                    'max_wait_time': metrics.mean.max_wait_time,
                    'throughput': metrics.mean.throughput,
                    'avg_queue_length': metrics.mean.avg_queue_length,
                    'max_queue_length': metrics.mean.max_queue_length,
                    'intersection_utilization': metrics.mean.intersection_utilization,
                    'fairness_score': metrics.mean.fairness_score,
                    'phase_switches': metrics.mean.phase_switches,
                    'phase_switch_frequency': metrics.mean.phase_switch_frequency,
                    'starvation_events': metrics.mean.starvation_events,
                    'max_wait_per_direction': metrics.mean.max_wait_per_direction
                },
                'std': metrics.std,
                'n_episodes': metrics.n_episodes
            }
        
        # Calculate improvements
        metrics_calc = MetricsCalculator()
        
        rl_metrics = scenario_metrics.get('IntelliLight-RL')
        mp_metrics = scenario_metrics.get('Max-Pressure')
        ft_metrics = scenario_metrics.get('Fixed-Timer')
        
        improvements_vs_mp = metrics_calc.compare_metrics(mp_metrics.mean, rl_metrics.mean) if mp_metrics and rl_metrics else {}
        improvements_vs_ft = metrics_calc.compare_metrics(ft_metrics.mean, rl_metrics.mean) if ft_metrics and rl_metrics else {}
        
        scenario_results['improvements'] = {
            'vs_max_pressure': improvements_vs_mp,
            'vs_fixed_timer': improvements_vs_ft
        }
        
        all_results['results_by_scenario'][scenario] = scenario_results
        
        # Print comparison table
        print(f"\n{'='*70}")
        print(f"COMPARISON: {scenario}")
        print(f"{'='*70}")
        print_comparison_table(scenario_results)
    
    # Print overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print_overall_summary(all_results)
    
    # Save results to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n Results saved to: {output_file}")
    
    # Cleanup
    engine.close()
    
    print(f"\n{'='*70}")
    print(f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    return all_results


def print_comparison_table(scenario_results: Dict):
    """Print formatted comparison table."""
    rl = scenario_results['IntelliLight-RL']['mean']
    
    print(f"\n{'Metric':<25} {'IntelliLight-RL':<15}")
    print("-" * 45)
    
    # Wait time
    print(f"{'Avg Wait Time (s)':<25} {rl['avg_wait_time']:>10.1f}")
    
    # Throughput
    print(f"{'Throughput (vehicles)':<25} {rl['throughput']:>10.0f}")
    
    # Queue
    print(f"{'Avg Queue Length':<25} {rl['avg_queue_length']:>10.1f}")
    
    # Utilization
    print(f"{'Utilization (%)':<25} {rl['intersection_utilization']*100:>10.1f}")
    
    # Safety metrics (RL only)
    print(f"\n{'Safety Metrics (RL only)':}")
    print(f"  Phase Switches: {rl['phase_switches']:.0f}")
    print(f"  Switch Frequency: {rl['phase_switch_frequency']:.1f} /min")
    print(f"  Starvation Events: {rl['starvation_events']:.0f}")
    print(f"  Max Wait (any direction): {rl['max_wait_time']:.1f}s")
    
    # Per-direction max wait
    print(f"\n{'Max Wait Per Direction (RL)':}")
    for direction, max_wait in rl['max_wait_per_direction'].items():
        print(f"  {direction}: {max_wait:.1f}s")
    
    # Improvements
    imp_mp = scenario_results['improvements']['vs_max_pressure']
    imp_ft = scenario_results['improvements']['vs_fixed_timer']
    
    print(f"\n{'RL Improvements':}")
    print(f"  vs Max-Pressure:")
    print(f"    Wait Time: {imp_mp.get('wait_time', 0):+.1f}%")
    print(f"    Throughput: {imp_mp.get('throughput', 0):+.1f}%")
    print(f"    Queue Length: {imp_mp.get('queue_length', 0):+.1f}%")
    
    print(f"  vs Fixed-Timer:")
    print(f"    Wait Time: {imp_ft.get('wait_time', 0):+.1f}%")
    print(f"    Throughput: {imp_ft.get('throughput', 0):+.1f}%")
    print(f"    Queue Length: {imp_ft.get('queue_length', 0):+.1f}%")


def print_overall_summary(all_results: Dict):
    """Print overall summary across all scenarios."""
    scenarios = all_results['scenarios']
    
    # Collect metrics across scenarios
    rl_wait_times = []
    rl_throughputs = []
    
    for scenario in scenarios:
        results = all_results['results_by_scenario'][scenario]
        rl_wait_times.append(results['IntelliLight-RL']['mean']['avg_wait_time'])
        rl_throughputs.append(results['IntelliLight-RL']['mean']['throughput'])
    
    # Calculate averages
    import numpy as np
    avg_rl_wait = np.mean(rl_wait_times)
    avg_rl_throughput = np.mean(rl_throughputs)
    
    print(f"\nAverage Performance (across all scenarios):")
    print(f"\n{'Metric':<25} {'IntelliLight-RL':<15}")
    print("-" * 45)
    print(f"{'Avg Wait Time (s)':<25} {avg_rl_wait:>10.1f}")
    print(f"{'Throughput (vehicles)':<25} {avg_rl_throughput:>10.0f}")
    
    # Overall verdict
    print(f"\n Overall Performance:")
    print("   RL controller evaluated successfully.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained IntelliLight model"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/checkpoints/intellilight_final.zip',
        help='Path to trained model (.zip file)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Number of episodes per scenario (default: 20)'
    )
    
    parser.add_argument(
        '--scenarios',
        nargs='+',
        default=None,
        help='Scenarios to test (default: all)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output JSON file path'
    )
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Show SUMO GUI (slow, for demo)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(
        model_path=args.model,
        n_episodes=args.episodes,
        scenarios=args.scenarios,
        output_file=args.output,
        use_gui=args.gui
    )


if __name__ == "__main__":
    main()