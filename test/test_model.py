"""
Headless smoke test: verifies the MuJoCo model loads and the Gym env steps.
No viewer required — suitable for CI.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.balancing_robot_env import TwoWheeledBalanceEnv


def test_env_steps(n_steps: int = 100):
    env = TwoWheeledBalanceEnv(render_mode=None)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (7,), f"expected obs shape (7,), got {obs.shape}"

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert "pitch" in info
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print(f"OK — {n_steps} steps, last pitch={info['pitch']:.3f}, reward={reward:.3f}")


if __name__ == "__main__":
    test_env_steps()
