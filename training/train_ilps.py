"""
ILPS Training Script
=====================

Training script for the Independent Learning with Parameter Sharing (ILPS)
per-phase traffic signal controller.

Key differences from centralized corridor training (train_rl_4phase.py):
  - Uses PerPhaseCorridorEnv (Discrete(8) action, Box(20,) obs)
  - 12 agent steps per cycle (3 junctions × 4 phases) vs 1 step per cycle
  - Green-wave coordination bonus in reward
  - Scalable: model trained on N junctions deploys to any M junctions

Usage (via main.py):
    python main.py train-ilps --steps 200000 --envs 4 --device auto

Direct usage:
    python training/train_ilps.py --timesteps 200000 --n-envs 4
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback
)

from rl.per_phase_env import PerPhaseCorridorEnv
from configs.parameters import training, paths, safety, ilps


# ── Callbacks ─────────────────────────────────────────────────────────────

class ILPSCurriculumCallback(BaseCallback):
    """
    Curriculum learning callback for ILPS per-phase environment.

    Progressively increases difficulty:
    - Stage 0 (0-66K steps):   WEEKEND traffic (light, single junction mastery)
    - Stage 1 (66K-133K):     EVENING_RUSH (medium, coordination learning)
    - Stage 2 (133K+ steps):  MORNING_RUSH (heavy + incidents)

    Note: With per-phase stepping, 200K timesteps covers far more decision
    points than the centralized model (12 steps/cycle vs 1 step/cycle).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_stage = 0
        self.stage_thresholds = [
            {"stage": 0, "until_step": 66000, "scenario": "WEEKEND"},
            {"stage": 1, "until_step": 133000, "scenario": "EVENING_RUSH"},
            {"stage": 2, "until_step": 200000, "scenario": "MORNING_RUSH"}
        ]

    def _on_step(self) -> bool:
        for threshold in self.stage_thresholds:
            if (self.num_timesteps >= threshold["until_step"]
                    and self.current_stage < threshold["stage"]):
                self.current_stage = threshold["stage"]

                if self.verbose > 0:
                    print(f"\n{'='*70}")
                    print(f"  ILPS CURRICULUM STAGE {self.current_stage}: "
                          f"{threshold['scenario']}")
                    print(f"{'='*70}\n")

                try:
                    self.training_env.env_method(
                        'set_curriculum_stage', self.current_stage
                    )
                except Exception as e:
                    print(f"  Warning: Could not update curriculum: {e}")

        return True


class ILPSSafetyCallback(BaseCallback):
    """
    Monitor safety violations during ILPS training.

    Tracks starvation events and logs per-junction metrics
    from the per-phase execution model.
    """

    def __init__(self, check_freq=2000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.safety_log = []
        self.execution_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                infos = self.locals.get('infos', [{}])
                for info in infos:
                    if info.get('is_execution_step', False):
                        self.execution_count += 1

                # Check for safety issues in the first env's info
                info = infos[0] if infos else {}
                if 'wait_times' in info:
                    max_wait = max(info['wait_times'])
                    if max_wait > safety.MAX_ACCEPTABLE_WAIT:
                        self.safety_log.append({
                            'step': self.num_timesteps,
                            'max_wait': max_wait,
                        })
                        if self.verbose > 0:
                            print(f"\n  Safety @ step {self.num_timesteps}: "
                                  f"max_wait={max_wait:.1f}s")

            except Exception:
                pass

        return True


class ILPSProgressCallback(BaseCallback):
    """
    Enhanced progress logging for ILPS training.

    Reports per-phase metrics and scalability info.
    """

    def __init__(self, print_freq=4000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = datetime.now()

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            elapsed = datetime.now() - self.start_time

            print(f"  Step {self.num_timesteps:>7,} | "
                  f"Elapsed: {str(elapsed).split('.')[0]} | "
                  f"FPS: {self.num_timesteps / max(elapsed.total_seconds(), 1):.0f}")

        return True


# ── Environment Factory ───────────────────────────────────────────────────

def make_ilps_env(
    rank: int,
    n_junctions: int = 3,
    curriculum_stage: int = 0,
    use_gui: bool = False,
):
    """
    Factory function returning a callable that creates one PerPhaseCorridorEnv.

    Args:
        rank:             Env index (0-based) for seeding
        n_junctions:      Number of corridor junctions
        curriculum_stage: Starting curriculum difficulty
        use_gui:          SUMO GUI (only rank 0, debugging only)

    Returns:
        Callable[[], PerPhaseCorridorEnv]
    """
    def _init():
        env = PerPhaseCorridorEnv(
            n_junctions=n_junctions,
            use_gui=(use_gui and rank == 0),
            episode_length=1800,
            curriculum_stage=curriculum_stage,
            seed=rank,
        )
        return env
    return _init


# ── Training Function ─────────────────────────────────────────────────────

def train_ilps(
    total_timesteps=200000,
    n_envs=4,
    n_junctions=3,
    learning_rate=3e-4,
    curriculum=True,
    save_freq=25000,
    checkpoint_dir=None,
    tensorboard_log=None,
    verbose=1,
    use_gui=False,
    resume_from=None,
    device="auto",
    use_subproc=False,
):
    """
    Train ILPS per-phase traffic signal controller.

    The trained model is a universal junction brain:
      - Input:  Box(20,) — local junction state + neighbor info
      - Output: Discrete(8) — duration index for one phase

    This model can be deployed to ANY number of junctions without
    retraining, because it never sees junction count or position.

    Args:
        total_timesteps: Total training steps
        n_envs:          Parallel environments
        n_junctions:     Junctions per corridor (default 3)
        learning_rate:   PPO learning rate
        curriculum:      Use curriculum learning
        save_freq:       Checkpoint interval
        checkpoint_dir:  Checkpoint directory
        tensorboard_log: TensorBoard log directory
        verbose:         Verbosity (0-2)
        use_gui:         SUMO GUI (debugging only)
        resume_from:     Checkpoint path to resume from
        device:          PyTorch device ('auto', 'cpu', 'cuda')
        use_subproc:     Use SubprocVecEnv (local only, not Colab)
    """
    # Directories
    if checkpoint_dir is None:
        checkpoint_dir = paths.CHECKPOINT_DIR
    if tensorboard_log is None:
        tensorboard_log = paths.TENSORBOARD_LOG

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    # Print config
    print("=" * 70)
    print("  INTELLILIGHT ILPS TRAINING")
    print("=" * 70)
    print(f"\n  Start:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Timesteps:   {total_timesteps:,}")
    print(f"  Envs:        {n_envs}")
    print(f"  Junctions:   {n_junctions}")
    print(f"  LR:          {learning_rate}")
    print(f"  Curriculum:  {curriculum}")
    print(f"  Device:      {device}")
    print(f"  Action:      Discrete(8) — per-phase ILPS")
    print(f"  Obs:         Box(20,) — 14 local + 6 neighbor/topology")
    print(f"  Steps/cycle: {n_junctions * 4} "
          f"({n_junctions} junctions × 4 phases)")
    print("=" * 70)

    # Create vectorized environment
    print("\n  Creating environments...")

    env_fns = [
        make_ilps_env(i, n_junctions=n_junctions, use_gui=use_gui)
        for i in range(n_envs)
    ]

    if use_subproc:
        print("  Using SubprocVecEnv (local mode)")
        env = SubprocVecEnv(env_fns)
    else:
        print("  Using DummyVecEnv (safe sequential mode)")
        env = DummyVecEnv(env_fns)

    env = VecMonitor(env)
    print(f"  {n_envs} environments created")

    # Create or load PPO model
    print("\n  Creating PPO model...")

    if resume_from:
        print(f"  Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env, device=device)
        model.learning_rate = learning_rate
        print("  Model loaded, ready to continue")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=training.N_STEPS,
            batch_size=training.BATCH_SIZE,
            n_epochs=training.N_EPOCHS,
            gamma=training.GAMMA,
            gae_lambda=training.GAE_LAMBDA,
            clip_range=training.CLIP_RANGE,
            ent_coef=training.ENT_COEF,
            max_grad_norm=training.MAX_GRAD_NORM,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            device=device,
        )
        print(f"  PPO on device: {model.device}")

    print(f"  Policy:  MlpPolicy")
    print(f"  Obs:     {env.observation_space}")
    print(f"  Action:  {env.action_space}")

    # Callbacks
    print("\n  Configuring callbacks...")
    callbacks = []

    # 1. Checkpoints
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=checkpoint_dir,
        name_prefix="intellilight_ilps",
        verbose=1,
    )
    callbacks.append(checkpoint_cb)

    # 2. Curriculum
    if curriculum:
        curriculum_cb = ILPSCurriculumCallback(verbose=1)
        callbacks.append(curriculum_cb)

    # 3. Safety monitor
    safety_cb = ILPSSafetyCallback(check_freq=2000, verbose=1)
    callbacks.append(safety_cb)

    # 4. Progress
    progress_cb = ILPSProgressCallback(print_freq=4000, verbose=1)
    callbacks.append(progress_cb)

    callback_list = CallbackList(callbacks)
    print(f"  {len(callbacks)} callbacks ready")

    # Train
    print("\n  TRAINING STARTED")
    print("=" * 70 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=training.LOG_INTERVAL,
            tb_log_name="ilps_run",
        )

    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")

    except Exception as e:
        print(f"\n\n  Training failed: {e}")
        raise

    finally:
        # Save final model
        final_path = Path(checkpoint_dir) / "intellilight_ilps_final.zip"
        model.save(final_path)

        print("\n" + "=" * 70)
        print("  ILPS TRAINING COMPLETE")
        print("=" * 70)
        print(f"  End:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Final model: {final_path}")

        # Safety summary
        if safety_cb.safety_log:
            print(f"\n  Safety violations: {len(safety_cb.safety_log)}")
        else:
            print("\n  No safety violations detected!")

        print(f"  Phase executions tracked: {safety_cb.execution_count}")

        print("\n  SCALABILITY NOTE:")
        print(f"  This model was trained on {n_junctions} junctions.")
        print(f"  It can be deployed to ANY number of junctions without")
        print(f"  retraining (obs=Box(20,), action=Discrete(8)).")
        print("=" * 70 + "\n")

        env.close()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for ILPS training."""
    parser = argparse.ArgumentParser(
        description="Train ILPS per-phase traffic controller"
    )

    parser.add_argument(
        '--timesteps', type=int, default=200000,
        help='Total training timesteps (default: 200000)'
    )
    parser.add_argument(
        '--n-envs', type=int, default=4,
        help='Parallel environments (default: 4)'
    )
    parser.add_argument(
        '--n-junctions', type=int, default=3,
        help='Junctions per corridor (default: 3)'
    )
    parser.add_argument(
        '--lr', type=float, default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    parser.add_argument(
        '--no-curriculum', action='store_true',
        help='Disable curriculum learning'
    )
    parser.add_argument(
        '--save-freq', type=int, default=25000,
        help='Checkpoint frequency (default: 25000)'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--verbose', type=int, default=1, choices=[0, 1, 2],
        help='Verbosity (default: 1)'
    )
    parser.add_argument(
        '--gui', action='store_true',
        help='SUMO GUI (slow, debugging only)'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Checkpoint path to resume from'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device (default: auto)'
    )
    parser.add_argument(
        '--subproc', action='store_true',
        help='Use SubprocVecEnv for true parallelism'
    )

    args = parser.parse_args()

    train_ilps(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        n_junctions=args.n_junctions,
        learning_rate=args.lr,
        curriculum=not args.no_curriculum,
        save_freq=args.save_freq,
        checkpoint_dir=args.checkpoint_dir,
        verbose=args.verbose,
        use_gui=args.gui,
        resume_from=args.resume,
        device=args.device,
        use_subproc=args.subproc,
    )


if __name__ == "__main__":
    main()
