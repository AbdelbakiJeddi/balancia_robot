import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from envs.balancing_robot_env import TwoWheeledBalanceEnv  # noqa: E402
MODEL_PATH = PROJECT_ROOT / "trained_models" / "ppo_balance_latest.zip"

# How many env steps to watch before checking disk for a newer checkpoint.
REFRESH_EVERY_STEPS = 200


def watch_model(model, model_path: Path, last_mtime: float) -> float:
    """Run the env with `model` until interrupted or a newer checkpoint appears.

    Returns the mtime that should be treated as "last seen" when this call
    returns (either unchanged, on Ctrl+C, or the newer one that triggered a
    reload).
    """
    env = TwoWheeledBalanceEnv(
        render_mode="human",
        randomize_target_vel=False,
    )

    obs, info = env.reset()
    steps_since_check = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()

            steps_since_check += 1
            if steps_since_check >= REFRESH_EVERY_STEPS:
                steps_since_check = 0

                if model_path.exists():
                    mtime = model_path.stat().st_mtime
                    if mtime != last_mtime:
                        print("Newer checkpoint found, reloading...")
                        return mtime

    except KeyboardInterrupt:
        print("Viewer stopped.")
        return last_mtime

    finally:
        env.close()


def load_model_safely(model_path: Path):
    """Try to load the checkpoint, tolerating a half-written file.

    If training writes the zip non-atomically, we might catch it mid-write.
    Retry briefly before giving up.
    """
    attempts = 5
    delay = 0.5

    for attempt in range(1, attempts + 1):
        try:
            return PPO.load(str(model_path))
        except Exception as e:
            if attempt == attempts:
                raise
            print(f"Load failed ({e}); retrying in {delay}s...")
            time.sleep(delay)


def main():
    print(f"Watching: {MODEL_PATH}")

    last_mtime = None

    while True:
        if MODEL_PATH.exists():
            mtime = MODEL_PATH.stat().st_mtime

            if mtime != last_mtime:
                print("Loading latest PPO model...")
                model = load_model_safely(MODEL_PATH)
                last_mtime = watch_model(model, MODEL_PATH, mtime)

        else:
            print("Waiting for PPO checkpoint...")

        time.sleep(1)


if __name__ == "__main__":
    main()