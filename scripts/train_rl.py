"""
Train PPO to balance the two-wheeled robot.

Usage:
    python train_ppo.py                     # train from scratch
    python train_ppo.py --timesteps 2000000 # longer run
    python train_ppo.py --resume            # continue from the last checkpoint

Install deps first:
    pip install "stable-baselines3[extra]" gymnasium mujoco
"""

import argparse
import sys
from pathlib import Path

# Make the sibling "envs" package importable regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.balancing_robot_env import TwoWheeledBalanceEnv  # noqa: E402

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


MODEL_DIR = PROJECT_ROOT / "trained_models"
LOG_DIR = PROJECT_ROOT / "logs"


def make_env():
    def _init():
        env = TwoWheeledBalanceEnv(render_mode=None, randomize_target_vel=False)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8,
                         help="Parallel environments; lower this if your CPU can't keep up.")
    parser.add_argument("--resume", action="store_true",
                         help="Load trained_models/ppo_balance_latest.zip and keep training.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(make_env(), n_envs=args.n_envs)

    latest_path = MODEL_DIR / "ppo_balance_latest.zip"

    if args.resume and latest_path.exists():
        print(f"Resuming from {latest_path}")
        model = PPO.load(str(latest_path), env=vec_env, tensorboard_log=str(LOG_DIR))
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            tensorboard_log=str(LOG_DIR),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(MODEL_DIR / "checkpoints"),
        name_prefix="ppo_balance",
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=not args.resume,
        tb_log_name="ppo_balance",
    )

    model.save(str(latest_path))
    print(f"\nTraining done. Saved to {latest_path}")
    print(f"View training curves with:\n  tensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    main()