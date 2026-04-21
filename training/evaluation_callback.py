"""
Evaluation Callback for Training
=================================

Integrates evaluation into the training loop.

Periodically evaluates the RL agent against baseline controllers
and logs results to TensorBoard for monitoring learning progress.
"""

import numpy as np
import logging
from typing import Optional, List
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from rl.traffic_env import TrafficEnv
from training.baseline_controllers import MaxPressureController, TrafficController
from training.evaluation_engine import EvaluationEngine
from training.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class EvaluationCallback(BaseCallback):
    """
    Callback for evaluating RL agent during training.
    
    Every N timesteps:
    1. Pauses training
    2. Runs evaluation episodes (RL vs baseline)
    3. Logs metrics to TensorBoard
    4. Resumes training
    
    This allows monitoring learning progress in real-time.
    """
    
    def __init__(
        self,
        eval_freq: int = 10000,
        n_eval_episodes: int = 3,
        scenarios: Optional[List[str]] = None,
        baseline_controller: Optional[TrafficController] = None,
        deterministic: bool = True,
        verbose: int = 1
    ):
        """
        Initialize evaluation callback.
        
        Args:
            eval_freq: Evaluate every N training timesteps
            n_eval_episodes: Episodes per scenario per controller
            scenarios: List of scenarios to test (defaults to all 3)
            baseline_controller: Baseline for comparison (defaults to Max-Pressure)
            deterministic: Use deterministic policy for RL
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.scenarios = scenarios or ["MORNING_RUSH", "EVENING_RUSH", "WEEKEND"]
        self.deterministic = deterministic
        
        # Create baseline controller
        self.baseline = baseline_controller or MaxPressureController(cyclic_mode=True)
        
        # Evaluation components (created on first call)
        self.eval_env = None
        self.eval_engine = None
        self.metrics_calc = None
        
        # Track last evaluation
        self.last_eval_timestep = 0
        
        logger.info(
            f"EvaluationCallback initialized: "
            f"freq={eval_freq}, episodes={n_eval_episodes}, "
            f"scenarios={scenarios}"
        )
    
    def _init_callback(self) -> None:
        """Initialize evaluation components on first call."""
        # Create separate evaluation environment (deterministic)
        self.eval_env = TrafficEnv(use_gui=False, curriculum_stage=0)
        
        # Create evaluation engine
        self.metrics_calc = MetricsCalculator(starvation_threshold=90)
        self.eval_engine = EvaluationEngine(
            env=self.eval_env,
            metrics_calculator=self.metrics_calc
        )
        
        logger.info("Evaluation components initialized")
    
    def _on_step(self) -> bool:
        """
        Called at each training step.
        
        Checks if it's time to evaluate and runs evaluation if needed.
        
        Returns:
            bool: True to continue training, False to stop
        """
        # Check if it's time to evaluate
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True  # Continue training
        
        # Time to evaluate!
        self.last_eval_timestep = self.num_timesteps
        
        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print(f" EVALUATION at {self.num_timesteps} timesteps")
            print(f"{'='*70}")
        
        # Run evaluation
        self._run_evaluation()
        
        if self.verbose >= 1:
            print(f"{'='*70}\n")
        
        return True  # Continue training
    
    def _run_evaluation(self):
        """Run evaluation across all scenarios."""
        # Ensure eval components are initialized
        if self.eval_engine is None:
            self._init_callback()
        
        # Create RL controller wrapper
        rl_controller = RLControllerWrapper(
            self.model,
            deterministic=self.deterministic
        )
        
        # Evaluate across all scenarios
        for scenario in self.scenarios:
            if self.verbose >= 1:
                print(f"\n Scenario: {scenario}")
            
            # Evaluate baseline
            if self.verbose >= 2:
                print(f"   {self.baseline.get_name()}:")
            
            baseline_metrics = self.eval_engine.evaluate_controller(
                self.baseline,
                n_episodes=self.n_eval_episodes,
                scenario=scenario,
                verbose=(self.verbose >= 2)
            )
            
            # Evaluate RL
            if self.verbose >= 2:
                print(f"   IntelliLight-RL:")
            
            rl_metrics = self.eval_engine.evaluate_controller(
                rl_controller,
                n_episodes=self.n_eval_episodes,
                scenario=scenario,
                verbose=(self.verbose >= 2)
            )
            
            # Calculate improvements
            improvements = self.metrics_calc.compare_metrics(
                baseline_metrics.mean,
                rl_metrics.mean
            )
            
            # Log to TensorBoard
            self._log_scenario_results(
                scenario,
                baseline_metrics,
                rl_metrics,
                improvements
            )
            
            # Print summary
            if self.verbose >= 1:
                self._print_summary(scenario, baseline_metrics, rl_metrics, improvements)
    
    def _log_scenario_results(
        self,
        scenario: str,
        baseline_metrics,
        rl_metrics,
        improvements: dict
    ):
        """Log evaluation results to TensorBoard."""
        prefix = f"eval/{scenario}"
        
        # Baseline metrics
        self.logger.record(f"{prefix}/baseline_wait_time", baseline_metrics.mean.avg_wait_time)
        self.logger.record(f"{prefix}/baseline_throughput", baseline_metrics.mean.throughput)
        self.logger.record(f"{prefix}/baseline_queue", baseline_metrics.mean.avg_queue_length)
        self.logger.record(f"{prefix}/baseline_utilization", baseline_metrics.mean.intersection_utilization)
        
        # RL metrics
        self.logger.record(f"{prefix}/rl_wait_time", rl_metrics.mean.avg_wait_time)
        self.logger.record(f"{prefix}/rl_throughput", rl_metrics.mean.throughput)
        self.logger.record(f"{prefix}/rl_queue", rl_metrics.mean.avg_queue_length)
        self.logger.record(f"{prefix}/rl_utilization", rl_metrics.mean.intersection_utilization)
        
        # Safety metrics
        self.logger.record(f"{prefix}/rl_phase_switches", rl_metrics.mean.phase_switches)
        self.logger.record(f"{prefix}/rl_switch_frequency", rl_metrics.mean.phase_switch_frequency)
        self.logger.record(f"{prefix}/rl_starvation_events", rl_metrics.mean.starvation_events)
        self.logger.record(f"{prefix}/rl_max_wait", rl_metrics.mean.max_wait_time)
        
        # Per-direction max wait
        for direction, max_wait in rl_metrics.mean.max_wait_per_direction.items():
            self.logger.record(f"{prefix}/rl_max_wait_{direction}", max_wait)
        
        # Improvements
        for metric, improvement in improvements.items():
            self.logger.record(f"{prefix}/improvement_{metric}", improvement)
        
        # Overall improvement (average of key metrics)
        key_improvements = [
            improvements.get('wait_time', 0),
            improvements.get('throughput', 0),
            improvements.get('queue_length', 0)
        ]
        overall = np.mean(key_improvements)
        self.logger.record(f"{prefix}/improvement_overall", overall)
    
    def _print_summary(
        self,
        scenario: str,
        baseline_metrics,
        rl_metrics,
        improvements: dict
    ):
        """Print evaluation summary to console."""
        print(f"\n   Results:")
        print(f"   {'Metric':<20} {'Baseline':<15} {'RL':<15} {'Improvement':<12}")
        print(f"   {'-'*62}")
        
        # Wait time
        print(
            f"   {'Avg Wait Time':<20} "
            f"{baseline_metrics.mean.avg_wait_time:>10.1f}s     "
            f"{rl_metrics.mean.avg_wait_time:>10.1f}s     "
            f"{improvements.get('wait_time', 0):>+8.1f}%"
        )
        
        # Throughput
        print(
            f"   {'Throughput':<20} "
            f"{baseline_metrics.mean.throughput:>10.0f} veh   "
            f"{rl_metrics.mean.throughput:>10.0f} veh   "
            f"{improvements.get('throughput', 0):>+8.1f}%"
        )
        
        # Queue
        print(
            f"   {'Avg Queue':<20} "
            f"{baseline_metrics.mean.avg_queue_length:>10.1f} veh   "
            f"{rl_metrics.mean.avg_queue_length:>10.1f} veh   "
            f"{improvements.get('queue_length', 0):>+8.1f}%"
        )
        
        # Utilization
        print(
            f"   {'Utilization':<20} "
            f"{baseline_metrics.mean.intersection_utilization:>10.1%}      "
            f"{rl_metrics.mean.intersection_utilization:>10.1%}      "
            f"{improvements.get('utilization', 0):>+8.1f}%"
        )
        
        # Safety metrics (RL only)
        print(f"\n   Safety Metrics (RL):")
        print(f"   Phase Switches: {rl_metrics.mean.phase_switches:.0f}")
        print(f"   Switch Frequency: {rl_metrics.mean.phase_switch_frequency:.1f} /min")
        print(f"   Starvation Events: {rl_metrics.mean.starvation_events:.0f}")
        print(f"   Max Wait (any dir): {rl_metrics.mean.max_wait_time:.1f}s")


class RLControllerWrapper(TrafficController):
    """
    Wraps SB3 model to match TrafficController interface.
    
    Allows using trained RL model with EvaluationEngine.
    """
    
    def __init__(self, model, deterministic: bool = True):
        """
        Initialize RL controller wrapper.
        
        Args:
            model: Trained SB3 model (PPO, DQN, etc.)
            deterministic: Use deterministic policy
        """
        self.model = model
        self.deterministic = deterministic
    
    def select_action(self, observation, info=None):
        """Select action using RL policy."""
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return tuple(action)
    
    def reset(self):
        """Reset (no-op for RL)."""
        pass
    
    def get_name(self):
        """Return controller name."""
        return "IntelliLight-RL"


if __name__ == "__main__":
    """Test evaluation callback (simplified)."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Testing EvaluationCallback")
    print("=" * 60)
    
    print("\n EvaluationCallback module loaded successfully")
    print("\nTo use in training:")
    print("  from training.evaluation_callback import EvaluationCallback")
    print("  callback = EvaluationCallback(eval_freq=10000, n_eval_episodes=3)")
    print("  model.learn(timesteps=200000, callback=callback)")
    
    print("\n" + "=" * 60)