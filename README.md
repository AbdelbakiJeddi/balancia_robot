# Balancia Robot — Two-Wheeled Balancing with Deep RL

> PPO (Stable-Baselines3) + MuJoCo + Gymnasium — symmetric drive, velocity tracking, and recovery from 13° tilt.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![MuJoCo 3.13](https://img.shields.io/badge/mujoco-3.13-green)
![SB3 2.9](https://img.shields.io/badge/stable_baselines3-2.9-orange)
![License MIT](https://img.shields.io/badge/license-MIT-lightgrey)

**Demo** — *add your screen recording here* `assets/demo.mp4` or GIF:

```bash
python scripts/play_rl.py --model trained_models/ppo_balance_latest.zip --target-vel 0.3
```

---

## Highlights

- **2M+ timesteps**, 8 parallel envs, ~7000 FPS — converges to `ep_len_mean≈4958/5000` (full episode) and `ep_rew_mean≈4599` peak
- **7-dim observation**: `[pitch, pitch_rate, yaw_rate, v_forward, l_wheel_vel, r_wheel_vel, target_vel]` — symmetric torque `Box(1,)` `[-1,1]`
- **Reward shaping**: `+1 alive - 7.0*pitch² - 0.4*vel_error² - 0.05*action_rate² ... -10 fall`
- **Robust initialization**: ±13° tilt + ±0.5 rad/s pitch rate, target_vel ramps `0.3 m/s²` and resamples every 300 steps (`MAX_TARGET_VEL=0.5 m/s`)

## Repo Structure

```
balancia_robot/
├── assets/two_wheeled.xml          # MuJoCo model (free joint base + 2 wheels)
├── envs/balancing_robot_env.py     # Gymnasium env (symmetric drive)
├── scripts/
│   ├── train_rl.py                 # PPO training (MlpPolicy, n_steps=2048)
│   └── play_rl.py                  # Watch policy in viewer
├── test/
│   ├── pid_balance.py              # Classical PID baseline (KP=9, KD=0.5)
│   ├── view_model.py               # Viewer for XML only
│   └── test_model.py
├── docs/ENV_REFERENCE.md           # Full env docs (obs/action/reward tables)
├── trained_models/                 # .zip checkpoints (gitignored, use Releases)
└── logs/                           # TensorBoard events (gitignored)
```

## Quick Start

```bash
git clone https://github.com/<you>/balancia-robot.git
cd balancia-robot

python -m venv env_mujoco && source env_mujoco/bin/activate
pip install -r requirements.txt
# requirements: gymnasium, mujoco, numpy, stable-baselines3, tensorboard

# Sanity-check model without RL
python test/view_model.py
python test/pid_balance.py  # click viewer, use ↑/↓ for speed

# Train
python scripts/train_rl.py --timesteps 1000000 --n-envs 8
tensorboard --logdir logs   # http://localhost:6006

# Resume
python scripts/train_rl.py --resume --timesteps 500000

# Play
python scripts/play_rl.py                               # latest, vel=0
python scripts/play_rl.py --model trained_models/checkpoints/ppo_balance_2015808_steps.zip --target-vel 0.3
```

## Training Details

| Param | Value |
|---|---|
| Policy | `MlpPolicy` |
| n_steps / batch | `2048` / `256` |
| lr / gamma / gae | `3e-4` / `0.99` / `0.95` |
| Checkpoint | every `50k // n_envs` steps |
| Logs | `logs/ppo_balance_*` |

Latest run `logs/ppo_balance_6` (2026-09-22): `16384 → 2031616` steps, `std 0.95→0.0085`, `explained_variance 0→0.97`, `value_loss 31973→4.4`.

Headless eval (20 seeds, 5000 steps, `target_vel=0`):
- `ppo_balance_latest.zip`: `12/20` falls (60% — overtrained)
- `ppo_balance_2015808_steps.zip`: `0/20` falls — **use this checkpoint**

## Environment

- **Action**: symmetric torque `left=right=torque*limit` (±5 N·m fallback)
- **Termination**: `|pitch|>35°` or `steps>=5000`
- **Velocity command**: `target_vel` ramps toward `goal~U(-0.5,0.5)` at `0.3 m/s²`, resampled every 300 steps when `randomize_target_vel=True` (`scripts/train_rl.py:35`)

See `docs/ENV_REFERENCE.md` for full tables.

## Results

> Replace with your plots: `rollout/ep_len_mean`, `rollout/ep_rew_mean`, `train/std`

- Balance-only (`target_vel=0`): recovers from ±13° and holds `pitch≈0.00±0.06 rad`
- Velocity tracking (`randomize=True`): follows `±0.5 m/s` with lean `≈0.15 rad` trade-off
- PID baseline comparison: RL smoother than `KP=9` classical controller

## Citation

```bibtex
@misc{balancia2026,
  title  = {Balancia Robot: Two-Wheeled Balancing with PPO},
  author = {Jeddi, Abdelbaki},
  year   = {2026},
  url    = {https://github.com/<you>/balancia-robot}
}
```

## License

MIT — see `LICENSE`.

---

**Tips**: Set `randomize_target_vel=False` in `scripts/train_rl.py:35` for balance-only training. Increase `PITCH_WEIGHT 7.0` if lean persists, or `ent_coef=0.01` to avoid `std` collapse.
