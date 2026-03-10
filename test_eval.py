import sys
sys.path.insert(0, 'C:/Users/kshit/cs/project/Intelli-Light')

from stable_baselines3 import PPO
from rl.traffic_env import TrafficEnv
from training.baseline_controllers import MaxPressureController, RLController
from training.evaluation_engine import EvaluationEngine

print("✓ All imports working!")

# Quick evaluation
print("\n🔧 Loading model...")
rl = RLController("models/checkpoints/intellilight_final.zip")
print("✓ Model loaded")

print("\n🔧 Creating baseline...")
baseline = MaxPressureController(cyclic_mode=True)
print("✓ Baseline created")

print("\n🔧 Running quick test (3 episodes)...")
engine = EvaluationEngine()

# Test RL
rl_metrics = engine.evaluate_controller(
    rl,
    n_episodes=3,
    scenario="WEEKEND",
    verbose=True
)

# Test baseline
baseline_metrics = engine.evaluate_controller(
    baseline,
    n_episodes=3,
    scenario="WEEKEND",
    verbose=True
)

print("\n" + "="*60)
print("QUICK COMPARISON:")
print("="*60)
print(f"\nRL Controller:")
print(f"  Throughput: {rl_metrics.mean.throughput:.0f}")
print(f"  Avg Wait: {rl_metrics.mean.avg_wait_time:.1f}s")
print(f"  Utilization: {rl_metrics.mean.intersection_utilization:.2%}")

print(f"\nMax-Pressure Baseline:")
print(f"  Throughput: {baseline_metrics.mean.throughput:.0f}")
print(f"  Avg Wait: {baseline_metrics.mean.avg_wait_time:.1f}s")
print(f"  Utilization: {baseline_metrics.mean.intersection_utilization:.2%}")

# Calculate improvement
from training.metrics_calculator import MetricsCalculator
calc = MetricsCalculator()
improvements = calc.compare_metrics(baseline_metrics.mean, rl_metrics.mean)

print(f"\nRL Improvements:")
print(f"  Wait Time: {improvements.get('wait_time', 0):+.1f}%")
print(f"  Throughput: {improvements.get('throughput', 0):+.1f}%")
print(f"  Utilization: {improvements.get('utilization', 0):+.1f}%")

engine.close()
print("\n✓ Evaluation complete!")