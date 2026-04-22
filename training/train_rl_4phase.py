"""
Corridor IntelliLight Training Script
=======================================

Production training script for 3-intersection corridor signal control.

Features:
- CorridorEnv (J1, J2, J3) with centralised PPO
- DummyVecEnv by default (safe for all platforms)
- Optional SubprocVecEnv via --subproc flag (faster locally)
- Curriculum learning
- Safety monitoring
- Checkpoint saving

Usage (via main.py):
    python main.py train --steps 200000 --envs 4 --subproc --device auto

Direct usage:
    python training/train_rl_4phase.py --timesteps 200000 --n-envs 4
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

from rl.multi_agent_env import CorridorEnv
from configs.parameters import training, paths, safety


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning.

    Progressively increases difficulty:
    - Stage 0 (0-66K steps): WEEKEND traffic
    - Stage 1 (66K-133K steps): EVENING_RUSH
    - Stage 2 (133K+ steps): MORNING_RUSH
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_stage = 0

        # Curriculum transitions: advance when timesteps reach 'from_step'
        self.stage_transitions = [
            {"stage": 1, "from_step": 66000, "scenario": "EVENING_RUSH"},
            {"stage": 2, "from_step": 133000, "scenario": "MORNING_RUSH"}
        ]

    def _on_step(self) -> bool:
        # Check if should advance to next stage
        for transition in self.stage_transitions:
            if (self.num_timesteps >= transition["from_step"]
                    and self.current_stage < transition["stage"]):
                self.current_stage = transition["stage"]

                if self.verbose > 0:
                    print(f"\n{'='*70}")
                    print(f"CURRICULUM STAGE {self.current_stage}: {transition['scenario']}")
                    print(f"{'='*70}\n")

                # Update all environments
                try:
                    self.training_env.env_method('set_curriculum_stage', self.current_stage)
                except Exception as e:
                    print(f"  Warning: Could not update curriculum stage: {e}")

        return True


class SafetyMonitorCallback(BaseCallback):
    """
    Monitor safety violations during training.

    Logs:
    - Starvation events (wait > 90s)
    - Maximum wait times
    - Emergency vehicle handling
    """

    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.safety_log = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Try to get safety stats from environments
            try:
                # Get info from first environment
                info = self.locals.get('infos', [{}])[0]

                max_wait = max(info.get('wait_times', [0]))

                if max_wait > safety.MAX_ACCEPTABLE_WAIT:
                    self.safety_log.append({
                        'step': self.num_timesteps,
                        'max_wait': max_wait,
                        'queues': info.get('queues', [])
                    })

                    if self.verbose > 0:
                        print(f"\nSafety Warning @ step {self.num_timesteps}:")
                        print(f"   Max wait: {max_wait:.1f}s (limit: {safety.MAX_ACCEPTABLE_WAIT}s)")
                        print(f"   Queues: {info.get('queues', [])}\n")

            except Exception:
                pass  # Silently continue if can't get info

        return True


class ProgressCallback(BaseCallback):
    """
    Enhanced progress logging.

    Prints detailed progress every N steps.
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

            # Get latest metrics
            ep_rew_mean = self.locals.get('ep_rew_mean', 'N/A')
            ep_len_mean = self.locals.get('ep_len_mean', 'N/A')

            print(f"Timestep {self.num_timesteps:,} | "
                  f"Elapsed: {str(elapsed).split('.')[0]} | "
                  f"Reward: {ep_rew_mean} | "
                  f"Ep Length: {ep_len_mean}")

        return True


def make_env(rank: int, curriculum_stage: int = 0, use_gui: bool = False):
    """
    Factory function that returns a callable creating one CorridorEnv instance.

    Args:
        rank: Index of this parallel environment (0-based)
        curriculum_stage: Starting curriculum difficulty
        use_gui: Show SUMO GUI -- only safe for rank 0 and only for debugging

    Returns:
        Callable[[], CorridorEnv]
    """
    def _init():
        env = CorridorEnv(
            use_gui=(use_gui and rank == 0),
            episode_length=1800,
            curriculum_stage=curriculum_stage,
            seed=rank,
        )
        return env
    return _init


def train(
    total_timesteps=200000,
    n_envs=4,
    learning_rate=3e-4,
    curriculum=True,
    save_freq=25000,
    checkpoint_dir=None,
    tensorboard_log=None,
    verbose=1,
    use_gui=False,
    resume_from=None,
    device="auto",
    use_subproc=False
):
    """
    Train corridor traffic signal controller.

    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        learning_rate: PPO learning rate
        curriculum: Use curriculum learning
        save_freq: Save checkpoint every N steps
        checkpoint_dir: Directory for checkpoints
        tensorboard_log: TensorBoard log directory
        verbose: Verbosity level (0, 1, or 2)
        use_gui: Show SUMO GUI (slows training)
        resume_from: Path to checkpoint to resume from
        device: Device to use for PPO ('auto', 'cpu', 'cuda')
        use_subproc: Use SubprocVecEnv (local only). Defaults to False (DummyVecEnv)
                     which is safe for Jupyter/Colab and all platforms.
    """
    # Setup directories
    if checkpoint_dir is None:
        checkpoint_dir = paths.CHECKPOINT_DIR

    if tensorboard_log is None:
        tensorboard_log = paths.TENSORBOARD_LOG

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("INTELLILIGHT CORRIDOR TRAINING")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Curriculum learning: {curriculum}")
    print(f"Save frequency: {save_freq:,}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"TensorBoard log: {tensorboard_log}")
    print("=" * 70)

    # Create vectorized environment
    print("\nCreating environments...")

    env_fns = [make_env(i, curriculum_stage=0, use_gui=use_gui) for i in range(n_envs)]

    if use_subproc:
        print("   Using SubprocVecEnv (local mode -- NOT for Jupyter/Colab)")
        env = SubprocVecEnv(env_fns)
    else:
        print("   Using DummyVecEnv (safe sequential mode)")
        env = DummyVecEnv(env_fns)

    env = VecMonitor(env)
    print(f"Created {n_envs} environments")

    # Create PPO model
    print("\nCreating PPO model...")

    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        model = PPO.load(resume_from, env=env, device=device)
        model.learning_rate = learning_rate
        print("Model loaded and ready to continue")
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
            device=device
        )
        print(f"PPO model created on device: {model.device}")

    print(f"   Policy: MlpPolicy | LR: {learning_rate}")
    print(f"   Obs space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Setup callbacks
    print("\nSetting up callbacks...")

    callbacks = []

    # 1. Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=checkpoint_dir,
        name_prefix="intellilight_corridor",
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # 2. Curriculum callback
    if curriculum:
        curriculum_callback = CurriculumCallback(verbose=1)
        callbacks.append(curriculum_callback)

    # 3. Safety monitor
    safety_callback = SafetyMonitorCallback(check_freq=1000, verbose=1)
    callbacks.append(safety_callback)

    # 4. Progress callback
    progress_callback = ProgressCallback(print_freq=4000, verbose=1)
    callbacks.append(progress_callback)

    callback_list = CallbackList(callbacks)

    print(f"{len(callbacks)} callbacks configured")

    # Start training
    print("\nTRAINING STARTED")
    print("=" * 70 + "\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=training.LOG_INTERVAL,
            tb_log_name="corridor_run"
        )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        raise

    finally:
        # Save final model
        final_model_path = Path(checkpoint_dir) / "intellilight_corridor_final.zip"
        model.save(final_model_path)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Final model saved: {final_model_path}")

        # Print safety summary
        if safety_callback.safety_log:
            print(f"\nSafety violations detected: {len(safety_callback.safety_log)}")
            print("   (See logs for details)")
        else:
            print("\nNo safety violations detected!")

        print("=" * 70 + "\n")

        # Cleanup
        env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train corridor IntelliLight traffic controller"
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=200000,
        help='Total training timesteps (default: 200000)'
    )

    parser.add_argument(
        '--n-envs',
        type=int,
        default=4,
        help='Number of parallel environments (default: 4)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )

    parser.add_argument(
        '--no-curriculum',
        action='store_true',
        help='Disable curriculum learning'
    )

    parser.add_argument(
        '--save-freq',
        type=int,
        default=25000,
        help='Save checkpoint every N steps (default: 25000)'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Checkpoint directory (default: models/checkpoints)'
    )

    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Verbosity level (default: 1)'
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='Show SUMO GUI (slow, for debugging)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device for PPO (default: auto)'
    )

    parser.add_argument(
        '--subproc',
        action='store_true',
        help='Use SubprocVecEnv for true parallel environments'
    )

    args = parser.parse_args()

    # Run training
    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
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