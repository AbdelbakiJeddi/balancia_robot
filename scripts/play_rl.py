"""
Load a trained PPO model and watch it balance the robot.

Usage:
    python watch_policy.py
    python watch_policy.py --model trained_models/checkpoints/ppo_balance_500000_steps.zip
"""

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.balancing_robot_env import TwoWheeledBalanceEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402


running = True


def handle_sigint(signum, frame):
    global running
    running = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "trained_models" / "ppo_balance_latest.zip"),
    )
    parser.add_argument("--target-vel", type=float, default=0.0,
                         help="Constant forward velocity command, in m/s.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_sigint)

    model = PPO.load(args.model)
    env = TwoWheeledBalanceEnv(render_mode="human")

    obs, _ = env.reset()
    env.target_vel = args.target_vel

    try:
        while running:
            step_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            print(
                f"\rpitch={info['pitch']:+6.3f} rad | "
                f"v={info['forward_velocity']:+5.2f} m/s | "
                f"reward={reward:+6.3f}   ",
                end="",
            )

            if terminated or truncated:
                print("\nEpisode ended, resetting...")
                obs, _ = env.reset()
                env.target_vel = args.target_vel

            remaining = env.dt - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    finally:
        print("\nClosing...")
        env.close()
        threading.Timer(2.0, lambda: os._exit(0)).start()


if __name__ == "__main__":
    main()