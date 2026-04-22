"""
Intelli-Light Unified CLI
==========================

Single entry point for all project operations.

Commands (Centralized — legacy):
    train        -- Train the corridor RL agent (centralized PPO)
    evaluate     -- Run evaluation against baselines
    test-gui     -- Visualise centralized model in SUMO GUI

Commands (ILPS — v2.0):
    train-ilps   -- Train ILPS per-phase universal junction brain
    eval-ilps    -- Evaluate ILPS model (with scalability test)
    test-ilps-gui -- Visualise ILPS model in SUMO GUI

Examples:
    # Lightweight ILPS sanity check
    python main.py train-ilps --steps 500 --envs 1

    # Full ILPS training on Colab
    python main.py train-ilps --steps 200000 --envs 4 --device auto

    # Evaluate ILPS model
    python main.py eval-ilps --model models/checkpoints/intellilight_ilps_final.zip

    # Scalability test: deploy 3-junction model to 9 junctions
    python main.py eval-ilps --model models/checkpoints/intellilight_ilps_final.zip --junctions 9

    # Legacy centralized training
    python main.py train --steps 200000 --envs 2
"""

import argparse
import sys
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


# ── Sub-command handlers ──────────────────────────────────────────────────────

def run_train(args):
    """Delegate to the corridor training script."""
    print("=" * 60)
    print("INTELLILIGHT — Corridor Training")
    print("=" * 60)
    # train() is the main function inside train_rl_4phase.py
    from training.train_rl_4phase import train
    train(
        total_timesteps=args.steps,
        n_envs=args.envs,
        device=args.device,
        use_subproc=args.subproc,
        resume_from=args.resume,
    )


def run_evaluate(args):
    """Run comprehensive multi-scenario evaluation against baselines."""
    print("=" * 60)
    print("INTELLILIGHT — Evaluation")
    print("=" * 60)
    from training.evaluate_model import evaluate_model
    evaluate_model(
        model_path=getattr(args, "model", "models/checkpoints/intellilight_corridor_final.zip")
    )


def run_test_gui(args):
    """
    Load a trained model and visualise one full episode in the SUMO GUI.
    Optionally plots queue and throughput histories with matplotlib.
    """
    print("=" * 60)
    print("INTELLILIGHT — GUI Visualisation")
    print("=" * 60)

    # Use CorridorEnv to match the trained model's action/obs spaces
    from rl.multi_agent_env import CorridorEnv

    try:
        model = PPO.load(args.model)
    except Exception as exc:
        print(f"[ERROR] Could not load model from '{args.model}': {exc}")
        sys.exit(1)

    env = CorridorEnv(use_gui=True, curriculum_stage=0)
    obs, info = env.reset()

    queue_history:      list = []
    throughput_history: list = []

    print(f"Model   : {args.model}")
    print(f"Env     : CorridorEnv (J1 → J2 → J3)")
    print("-" * 60)

    total_reward = 0.0
    steps = 0
    done = truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        total_queue = info.get("queues", 0)
        throughput  = info.get("throughput", 0)

        queue_history.append(total_queue)
        throughput_history.append(throughput)

        if steps % 10 == 0:
            print(
                f"Cycle {steps:4d} | reward={total_reward:8.2f} | "
                f"throughput={throughput:5d} | queue={total_queue:5.0f}"
            )

    env.close()
    print("-" * 60)
    print(f"Episode complete — cycles: {steps}, total reward: {total_reward:.2f}")

    if args.plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(queue_history)
        ax1.set_title("Total Corridor Queue Over Time")
        ax1.set_xlabel("Cycle"); ax1.set_ylabel("Halting Vehicles")

        ax2.plot(throughput_history)
        ax2.set_title("Cumulative Throughput")
        ax2.set_xlabel("Cycle"); ax2.set_ylabel("Vehicles Arrived")

        plt.tight_layout()
        plt.show()


# ── ILPS Sub-command handlers ─────────────────────────────────────────────────

def run_train_ilps(args):
    """Train ILPS per-phase universal junction brain."""
    print("=" * 60)
    print("INTELLILIGHT — ILPS Per-Phase Training")
    print("=" * 60)
    from training.train_ilps import train_ilps
    train_ilps(
        total_timesteps=args.steps,
        n_envs=args.envs,
        n_junctions=args.junctions,
        device=args.device,
        use_subproc=args.subproc,
        resume_from=args.resume,
    )


def run_eval_ilps(args):
    """Evaluate ILPS model with optional scalability test."""
    print("=" * 60)
    print("INTELLILIGHT — ILPS Evaluation")
    print("=" * 60)
    from training.evaluate_ilps import evaluate_ilps_model
    evaluate_ilps_model(
        model_path=args.model,
        n_episodes=args.episodes,
        n_junctions=args.junctions,
        output_file=args.output,
        use_gui=args.gui,
    )


def run_test_ilps_gui(args):
    """Visualise ILPS model on PerPhaseCorridorEnv with SUMO GUI."""
    print("=" * 60)
    print("INTELLILIGHT — ILPS GUI Visualisation")
    print("=" * 60)

    from rl.per_phase_env import PerPhaseCorridorEnv

    try:
        model = PPO.load(args.model)
    except Exception as exc:
        print(f"[ERROR] Could not load model from '{args.model}': {exc}")
        sys.exit(1)

    env = PerPhaseCorridorEnv(
        n_junctions=args.junctions, use_gui=True, curriculum_stage=0
    )
    obs, info = env.reset()

    print(f"Model      : {args.model}")
    print(f"Junctions  : {args.junctions}")
    print(f"Env        : PerPhaseCorridorEnv (ILPS)")
    print("-" * 60)

    total_reward = 0.0
    steps = 0
    done = truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1

        if info.get("is_execution_step", False) and steps % 3 == 0:
            print(
                f"Step {steps:4d} | J={info['junction_id']} "
                f"P={info['phase']} | reward={total_reward:8.2f} | "
                f"throughput={info.get('throughput', 0):5d}"
            )

    env.close()
    print("-" * 60)
    print(f"Episode complete — steps: {steps}, total reward: {total_reward:.2f}")


# ── CLI definition ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="intelli-light",
        description="Intelli-Light Unified CLI — corridor traffic signal RL",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ──
    train_p = subparsers.add_parser("train", help="Train the corridor RL agent")
    train_p.add_argument(
        "--steps", type=int, default=200000,
        help="Total training timesteps (default: 200 000)"
    )
    train_p.add_argument(
        "--envs", type=int, default=2,
        help="Parallel SUMO environments (default: 2, safe for RTX 3050)"
    )
    train_p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Device to run PPO on (default: auto)"
    )
    train_p.add_argument(
        "--subproc", action="store_true",
        help="Use SubprocVecEnv for true parallelism (faster locally)"
    )
    train_p.add_argument(
        "--resume", type=str, default=None,
        help="Path to saved .zip checkpoint to resume training from"
    )

    # ── evaluate ──
    eval_p = subparsers.add_parser("evaluate", help="Evaluate corridor model against baselines")
    eval_p.add_argument(
        "--model", type=str,
        default="models/checkpoints/intellilight_corridor_final.zip",
        help="Path to saved .zip checkpoint"
    )

    # ── test-gui ──
    test_p = subparsers.add_parser("test-gui", help="Visualise a trained model in SUMO GUI")
    test_p.add_argument(
        "--model", type=str,
        default="models/checkpoints/intellilight_corridor_final.zip",
        help="Path to saved .zip checkpoint"
    )
    test_p.add_argument(
        "--plot", action="store_true",
        help="Show matplotlib plots after episode ends"
    )

    # ── train-ilps ──
    ilps_train_p = subparsers.add_parser(
        "train-ilps", help="Train ILPS per-phase universal junction brain"
    )
    ilps_train_p.add_argument(
        "--steps", type=int, default=200000,
        help="Total training timesteps (default: 200000)"
    )
    ilps_train_p.add_argument(
        "--envs", type=int, default=4,
        help="Parallel SUMO environments (default: 4)"
    )
    ilps_train_p.add_argument(
        "--junctions", type=int, default=3,
        help="Junctions per corridor (default: 3)"
    )
    ilps_train_p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Device (default: auto)"
    )
    ilps_train_p.add_argument(
        "--subproc", action="store_true",
        help="Use SubprocVecEnv for true parallelism"
    )
    ilps_train_p.add_argument(
        "--resume", type=str, default=None,
        help="Resume from ILPS checkpoint"
    )

    # ── eval-ilps ──
    ilps_eval_p = subparsers.add_parser(
        "eval-ilps", help="Evaluate ILPS model (with scalability test)"
    )
    ilps_eval_p.add_argument(
        "--model", type=str,
        default="models/checkpoints/intellilight_ilps_final.zip",
        help="Path to trained ILPS model (.zip)"
    )
    ilps_eval_p.add_argument(
        "--episodes", type=int, default=5,
        help="Episodes per scenario (default: 5)"
    )
    ilps_eval_p.add_argument(
        "--junctions", type=int, default=3,
        help="Junctions (3=standard, 9=scalability test)"
    )
    ilps_eval_p.add_argument(
        "--output", type=str, default="ilps_evaluation_results.json",
        help="Output JSON file"
    )
    ilps_eval_p.add_argument(
        "--gui", action="store_true",
        help="Show SUMO GUI"
    )

    # ── test-ilps-gui ──
    ilps_gui_p = subparsers.add_parser(
        "test-ilps-gui", help="Visualise ILPS model in SUMO GUI"
    )
    ilps_gui_p.add_argument(
        "--model", type=str,
        default="models/checkpoints/intellilight_ilps_final.zip",
        help="Path to trained ILPS model (.zip)"
    )
    ilps_gui_p.add_argument(
        "--junctions", type=int, default=3,
        help="Junctions to deploy (default: 3)"
    )

    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "test-gui":
        run_test_gui(args)
    elif args.command == "train-ilps":
        run_train_ilps(args)
    elif args.command == "eval-ilps":
        run_eval_ilps(args)
    elif args.command == "test-ilps-gui":
        run_test_ilps_gui(args)
