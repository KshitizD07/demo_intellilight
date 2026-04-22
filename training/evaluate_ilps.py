"""
ILPS Model Evaluation Script
==============================

Evaluates a trained ILPS per-phase model on the PerPhaseCorridorEnv.

Features:
  - Run trained ILPS model on 3-junction corridor (default)
  - Scalability test: deploy same model to 9 junctions (zero retraining)
  - Scenario sweep: WEEKEND, EVENING_RUSH, MORNING_RUSH
  - Per-junction and aggregate metrics
  - Web-ready decision log export

Usage:
    python training/evaluate_ilps.py --model models/checkpoints/intellilight_ilps_final.zip
    python training/evaluate_ilps.py --model models/checkpoints/intellilight_ilps_final.zip --scalability 9
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from rl.per_phase_env import PerPhaseCorridorEnv


def evaluate_ilps_model(
    model_path: str,
    n_episodes: int = 5,
    n_junctions: int = 3,
    scenarios: List[str] = None,
    output_file: str = None,
    use_gui: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a trained ILPS model on PerPhaseCorridorEnv.

    Args:
        model_path:   Path to trained ILPS model (.zip)
        n_episodes:   Episodes per scenario
        n_junctions:  Number of junctions (3 = trained config, 9 = scalability test)
        scenarios:    List of scenarios to evaluate
        output_file:  Path to save JSON results
        use_gui:      SUMO GUI
        verbose:      Print detailed output

    Returns:
        Dict of evaluation results
    """
    if scenarios is None:
        scenarios = ["WEEKEND", "EVENING_RUSH", "MORNING_RUSH"]

    scenario_stage_map = {
        "WEEKEND": 0, "EVENING_RUSH": 1, "MORNING_RUSH": 2
    }

    # Load model
    print("=" * 70)
    print("  INTELLILIGHT ILPS EVALUATION")
    print("=" * 70)
    print(f"\n  Model:       {model_path}")
    print(f"  Junctions:   {n_junctions}")
    print(f"  Episodes:    {n_episodes} per scenario")
    print(f"  Scenarios:   {', '.join(scenarios)}")
    print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model = PPO.load(model_path)
    print(f"  Model loaded: obs={model.observation_space}, act={model.action_space}")
    print("=" * 70)

    all_results = {
        "model_path": model_path,
        "n_junctions": n_junctions,
        "n_episodes": n_episodes,
        "timestamp": datetime.now().isoformat(),
        "scenarios": {},
    }

    for scenario in scenarios:
        print(f"\n  --- {scenario} ---")
        stage = scenario_stage_map.get(scenario, 0)

        episode_metrics = []

        for ep in range(n_episodes):
            env = PerPhaseCorridorEnv(
                n_junctions=n_junctions,
                use_gui=use_gui,
                episode_length=1800,
                curriculum_stage=stage,
            )

            try:
                obs, info = env.reset()
                total_reward = 0.0
                steps = 0
                execution_steps = 0

                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(int(action))
                    total_reward += reward
                    steps += 1

                    if info.get("is_execution_step", False):
                        execution_steps += 1

                    if terminated or truncated:
                        break

                # Collect metrics
                metrics = {
                    "episode": ep,
                    "total_reward": total_reward,
                    "steps": steps,
                    "execution_steps": execution_steps,
                    "cycles": env.cycle_count,
                    "sim_time": env.simulation_step,
                    "throughput": env.cumulative_arrived,
                    "decisions": len(env.decision_log),
                }

                episode_metrics.append(metrics)

                if verbose:
                    print(f"    Ep {ep+1}/{n_episodes}: "
                          f"R={total_reward:+.1f}  "
                          f"T={env.cumulative_arrived}  "
                          f"Cycles={env.cycle_count}  "
                          f"Steps={steps}")

            except Exception as e:
                print(f"    Ep {ep+1} FAILED: {e}")
                episode_metrics.append({"episode": ep, "error": str(e)})

            finally:
                env.close()

        # Aggregate scenario metrics
        valid = [m for m in episode_metrics if "error" not in m]
        if valid:
            scenario_summary = {
                "n_episodes": len(valid),
                "mean_reward": float(np.mean([m["total_reward"] for m in valid])),
                "std_reward": float(np.std([m["total_reward"] for m in valid])),
                "mean_throughput": float(np.mean([m["throughput"] for m in valid])),
                "std_throughput": float(np.std([m["throughput"] for m in valid])),
                "mean_cycles": float(np.mean([m["cycles"] for m in valid])),
                "mean_steps": float(np.mean([m["steps"] for m in valid])),
                "episodes": episode_metrics,
            }
        else:
            scenario_summary = {"n_episodes": 0, "error": "All episodes failed"}

        all_results["scenarios"][scenario] = scenario_summary

        if valid:
            print(f"\n    Summary: R={scenario_summary['mean_reward']:+.1f} ± "
                  f"{scenario_summary['std_reward']:.1f}  "
                  f"Throughput={scenario_summary['mean_throughput']:.0f} ± "
                  f"{scenario_summary['std_throughput']:.0f}")

    # Print final summary
    print(f"\n{'='*70}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Scenario':<20} {'Reward':>10} {'Throughput':>12} {'Cycles':>8}")
    print(f"  {'-'*50}")

    for scenario in scenarios:
        s = all_results["scenarios"].get(scenario, {})
        if s.get("n_episodes", 0) > 0:
            print(f"  {scenario:<20} {s['mean_reward']:>+10.1f} "
                  f"{s['mean_throughput']:>12.0f} "
                  f"{s['mean_cycles']:>8.1f}")

    if n_junctions != 3:
        print(f"\n  SCALABILITY: Model trained on 3 junctions, "
              f"evaluated on {n_junctions} junctions")
        print(f"  Zero retraining — same Discrete(8) / Box(20,) policy")

    print(f"\n{'='*70}\n")

    # Save results
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Results saved to: {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained ILPS per-phase model"
    )

    parser.add_argument(
        "--model", type=str,
        default="models/checkpoints/intellilight_ilps_final.zip",
        help="Path to trained ILPS model (.zip)"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Episodes per scenario (default: 5)"
    )
    parser.add_argument(
        "--junctions", type=int, default=3,
        help="Number of junctions (default: 3, use 9 for scalability test)"
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Scenarios to test (default: all)"
    )
    parser.add_argument(
        "--output", type=str, default="ilps_evaluation_results.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Show SUMO GUI"
    )

    args = parser.parse_args()

    evaluate_ilps_model(
        model_path=args.model,
        n_episodes=args.episodes,
        n_junctions=args.junctions,
        scenarios=args.scenarios,
        output_file=args.output,
        use_gui=args.gui,
    )


if __name__ == "__main__":
    main()
