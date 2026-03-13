"""
IntelliLight Training Module
=============================

Professional training script for reinforcement learning traffic signal control.

Features:
- Command-line interface for easy configuration
- Automatic checkpointing and model saving
- TensorBoard integration for monitoring
- Curriculum learning support
- Resumable training from checkpoints
- Multi-environment support for faster training

Usage:
    python training/train_rl.py --timesteps 100000 --curriculum --save-freq 10000
    python training/train_rl.py --timesteps 50000 --gui --checkpoint models/checkpoints/model_50000.zip
    python training/train_rl.py --help
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl.traffic_env import TrafficEnv4Phase
from configs.parameters import TrainingConfig, Paths


class ProgressCallback(BaseCallback):
    """
    Custom callback for printing training progress to console.
    
    Provides real-time feedback during training including:
    - Episode rewards
    - Episode lengths
    - Timestep progress
    """
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        """
        Initialize progress callback.
        
        Args:
            check_freq: How often to print progress (in timesteps)
            verbose: Verbosity level (0=none, 1=info, 2=debug)
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.start_time = None
    
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.start_time = datetime.now()
        if self.verbose >= 1:
            print("\n" + "=" * 70)
            print("🚀 TRAINING STARTED")
            print("=" * 70)
            print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total timesteps: {self.locals.get('total_timesteps', 'unknown')}")
            print("=" * 70 + "\n")
    
    def _on_step(self) -> bool:
        """
        Called at each step.
        
        Returns:
            bool: If False, training is aborted
        """
        if self.n_calls % self.check_freq == 0:
            if self.verbose >= 1:
                # Calculate elapsed time
                elapsed = datetime.now() - self.start_time
                elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
                
                # Get episode stats if available
                ep_rew_mean = self.locals.get('ep_rew_mean', 'N/A')
                ep_len_mean = self.locals.get('ep_len_mean', 'N/A')
                
                print(f"⏱️  Timestep {self.num_timesteps:,} | "
                      f"Elapsed: {elapsed_str} | "
                      f"Reward: {ep_rew_mean if ep_rew_mean == 'N/A' else f'{ep_rew_mean:.2f}'} | "
                      f"Ep Length: {ep_len_mean if ep_len_mean == 'N/A' else f'{ep_len_mean:.1f}'}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.verbose >= 1:
            total_time = datetime.now() - self.start_time
            total_time_str = str(total_time).split('.')[0]
            
            print("\n" + "=" * 70)
            print("✅ TRAINING COMPLETED")
            print("=" * 70)
            print(f"Total time: {total_time_str}")
            print(f"Total timesteps: {self.num_timesteps:,}")
            print("=" * 70 + "\n")


class CurriculumCallback(BaseCallback):
    """
    Callback for automatic curriculum learning progression.
    
    Increases traffic difficulty as the agent improves.
    """
    
    def __init__(
        self,
        stage_transitions: list,
        verbose: int = 1
    ):
        """
        Initialize curriculum callback.
        
        Args:
            stage_transitions: List of timesteps at which to increase stage
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.stage_transitions = sorted(stage_transitions)
        self.current_stage = 0
    
    def _on_step(self) -> bool:
        """Check if we should progress to next curriculum stage."""
        # Check if we've reached a transition point
        if self.stage_transitions and self.num_timesteps >= self.stage_transitions[0]:
            self.current_stage += 1
            self.stage_transitions.pop(0)
            
            if self.verbose >= 1:
                print(f"\n{'=' * 70}")
                print(f"📈 CURRICULUM PROGRESSION: Stage {self.current_stage}")
                print(f"{'=' * 70}\n")
            
            # Update all environments to new stage
            try:
                self.training_env.env_method('set_curriculum_stage',self.current_stage)
            
            except AttributeError:
                try:
                    self.training_env.set_curriculum_stage(self.current_stage)
                except Exception as e:
                    print(f"⚠️  Warning: Could not set curriculum stage on environment: {e}")
                # Single environment
                
        
        return True


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train IntelliLight RL traffic control agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python training/train_rl.py --timesteps 100000
  
  # Training with GUI visualization
  python training/train_rl.py --timesteps 50000 --gui
  
  # Training with curriculum learning
  python training/train_rl.py --timesteps 300000 --curriculum
  
  # Resume from checkpoint
  python training/train_rl.py --timesteps 100000 --checkpoint models/checkpoints/model_50000.zip
  
  # Custom save frequency
  python training/train_rl.py --timesteps 200000 --save-freq 20000
  
  # Multiple parallel environments (faster)
  python training/train_rl.py --timesteps 100000 --n-envs 4
        """
    )
    
    # Training parameters
    parser.add_argument(
        '--timesteps',
        type=int,
        default=TrainingConfig.TOTAL_TIMESTEPS,
        help=f'Total training timesteps (default: {TrainingConfig.TOTAL_TIMESTEPS})'
    )
    
    parser.add_argument(
        '--save-freq',
        type=int,
        default=TrainingConfig.SAVE_FREQUENCY,
        help=f'Save checkpoint every N timesteps (default: {TrainingConfig.SAVE_FREQUENCY})'
    )
    
    # Environment settings
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Enable SUMO GUI visualization (slower but visible)'
    )
    
    parser.add_argument(
        '--n-envs',
        type=int,
        default=1,
        help='Number of parallel environments (default: 1, recommended: 4-8)'
    )
    
    # Curriculum learning
    parser.add_argument(
        '--curriculum',
        action='store_true',
        help='Enable curriculum learning (progressive difficulty)'
    )
    
    parser.add_argument(
        '--curriculum-stage',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Starting curriculum stage (0=easy, 1=medium, 2=hard, default: 0)'
    )
    
    # Model management
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='intellilight',
        help='Base name for saved models (default: intellilight)'
    )
    
    # Logging
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='TensorBoard log directory (default: logs/TIMESTAMP)'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Verbosity level (0=quiet, 1=info, 2=debug, default: 1)'
    )
    
    return parser.parse_args()


def create_log_directory(base_name: str = None) -> str:
    """
    Create timestamped log directory.
    
    Args:
        base_name: Optional base name for the log directory
    
    Returns:
        str: Path to created log directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name = f"{base_name}_{timestamp}" if base_name else timestamp
    log_dir = os.path.join(Paths.LOGS_DIR, log_name)
    
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir


def make_env(use_gui: bool = False, curriculum_stage: int = 0, rank: int = 0):
    """
    Create a single environment instance.
    
    Args:
        use_gui: Whether to use SUMO GUI
        curriculum_stage: Initial curriculum stage
        rank: Environment rank (for parallel envs)
    
    Returns:
        callable: Function that creates the environment
    """
    def _init():
        env = TrafficEnv4Phase(
            use_gui=use_gui and rank == 0,  # Only show GUI for first env
            curriculum_stage=curriculum_stage
        )
        env = Monitor(env)
        return env
    
    return _init


def create_vectorized_env(n_envs: int, use_gui: bool, curriculum_stage: int):
    """
    Create vectorized environment for parallel training.
    
    Args:
        n_envs: Number of parallel environments
        use_gui: Whether to use SUMO GUI (only for first env)
        curriculum_stage: Initial curriculum stage
    
    Returns:
        VecEnv: Vectorized environment
    """
    if n_envs == 1:
        # Single environment
        return DummyVecEnv([make_env(use_gui, curriculum_stage, 0)])
    else:
        # Multiple parallel environments
        env_fns = [make_env(use_gui, curriculum_stage, i) for i in range(n_envs)]
        return SubprocVecEnv(env_fns)


def setup_callbacks(
    save_freq: int,
    checkpoint_dir: str,
    model_name: str,
    enable_curriculum: bool,
    verbose: int
):
    """
    Setup training callbacks.
    
    Args:
        save_freq: Save frequency in timesteps
        checkpoint_dir: Directory for checkpoints
        model_name: Base name for saved models
        enable_curriculum: Whether to use curriculum learning
        verbose: Verbosity level
    
    Returns:
        CallbackList: Combined callbacks
    """
    callbacks = []
    
    # Checkpoint callback (saves model periodically)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix=model_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=verbose
    )
    callbacks.append(checkpoint_callback)
    
    # Progress callback (prints training info)
    progress_callback = ProgressCallback(
        check_freq=1000,
        verbose=verbose
    )
    callbacks.append(progress_callback)
    
    # Curriculum callback (if enabled)
    if enable_curriculum:
        from configs.parameters import CurriculumConfig
        curriculum_callback = CurriculumCallback(
            stage_transitions=CurriculumConfig.TRANSITIONS.copy(),
            verbose=verbose
        )
        callbacks.append(curriculum_callback)
    
    return CallbackList(callbacks)


def print_training_config(args, log_dir: str, checkpoint_dir: str):
    """
    Print training configuration to console.
    
    Args:
        args: Parsed command line arguments
        log_dir: TensorBoard log directory
        checkpoint_dir: Checkpoint save directory
    """
    print("\n" + "=" * 70)
    print("🎯 INTELLILIGHT TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Training timesteps:     {args.timesteps:,}")
    print(f"Checkpoint frequency:   Every {args.save_freq:,} steps")
    print(f"Parallel environments:  {args.n_envs}")
    print(f"SUMO GUI:               {'Enabled' if args.gui else 'Disabled'}")
    print(f"Curriculum learning:    {'Enabled' if args.curriculum else 'Disabled'}")
    if args.curriculum:
        print(f"  Starting stage:       {args.curriculum_stage}")
    if args.checkpoint:
        print(f"Resume from:            {args.checkpoint}")
    print(f"\nTensorBoard logs:       {log_dir}")
    print(f"Model checkpoints:      {checkpoint_dir}")
    print(f"Verbosity:              Level {args.verbose}")
    print("=" * 70 + "\n")


def train(args):
    """
    Main training function.
    
    Args:
        args: Parsed command line arguments
    """
    # Setup directories
    log_dir = args.log_dir or create_log_directory(args.model_name)
    checkpoint_dir = Paths.CHECKPOINTS_DIR
    
    # Print configuration
    print_training_config(args, log_dir, checkpoint_dir)
    
    # Create environment
    print("🔧 Initializing environment...")
    env = create_vectorized_env(
        n_envs=args.n_envs,
        use_gui=args.gui,
        curriculum_stage=args.curriculum_stage
    )
    print(f"✓ Environment initialized ({args.n_envs} parallel instance{'s' if args.n_envs > 1 else ''})")
    
    # Create or load model
    if args.checkpoint:
        print(f"\n📂 Loading checkpoint: {args.checkpoint}")
        model = PPO.load(
            args.checkpoint,
            env=env,
            verbose=args.verbose,
            tensorboard_log=log_dir
        )
        print("✓ Checkpoint loaded successfully")
    else:
        print("\n🤖 Creating new PPO model...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=TrainingConfig.LEARNING_RATE,
            n_steps=TrainingConfig.N_STEPS,
            batch_size=TrainingConfig.BATCH_SIZE,
            n_epochs=TrainingConfig.N_EPOCHS,
            gamma=TrainingConfig.GAMMA,
            gae_lambda=TrainingConfig.GAE_LAMBDA,
            clip_range=TrainingConfig.CLIP_RANGE,
            ent_coef=TrainingConfig.ENT_COEF,
            vf_coef=TrainingConfig.VF_COEF,
            max_grad_norm=TrainingConfig.MAX_GRAD_NORM,
            verbose=args.verbose,
            tensorboard_log=log_dir
        )
        print("✓ Model created successfully")
    
    # Setup callbacks
    callbacks = setup_callbacks(
        save_freq=args.save_freq,
        checkpoint_dir=checkpoint_dir,
        model_name=args.model_name,
        enable_curriculum=args.curriculum,
        verbose=args.verbose
    )
    
    # Train
    print("\n" + "=" * 70)
    print("🏋️  BEGINNING TRAINING")
    print("=" * 70)
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
    finally:
        # Save final model
        final_model_path = os.path.join(checkpoint_dir, f"{args.model_name}_final.zip")
        model.save(final_model_path)
        print(f"\n💾 Final model saved: {final_model_path}")
        
        # Clean up
        env.close()
        print("✓ Environment closed")
    
    # Print completion message
    print("\n" + "=" * 70)
    print("🎉 TRAINING SESSION COMPLETE!")
    print("=" * 70)
    print(f"\nTo view training metrics:")
    print(f"  tensorboard --logdir {log_dir}")
    print(f"\nTo resume training:")
    print(f"  python training/train_rl.py --checkpoint {final_model_path} --timesteps ADDITIONAL_STEPS")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        train(args)
    except Exception as e:
        print(f"\n❌ ERROR: Training failed with exception:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()