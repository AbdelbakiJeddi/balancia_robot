# BalancingRobotEnv — Environment Reference

A Gymnasium environment wrapping the two-wheeled MuJoCo model (`assets/two_wheeled.xml`)
for training a balancing/driving policy with RL (e.g. PPO via Stable-Baselines3).

```python
from envs.balancing_robot_env import BalancingRobotEnv

env = BalancingRobotEnv(render_mode=None, frame_skip=5)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

---

## Model & timing

| Setting | Value | Notes |
|---|---|---|
| Model file | `assets/two_wheeled.xml` | Loaded once in `__init__` |
| `frame_skip` | `5` (constructor arg) | Physics steps per `env.step()` call. The policy acts every 5th physics tick, not every tick — this is an action-repeat, common in MuJoCo RL tasks to make the effective control rate coarser and training faster. |
| `max_steps` | `1000` | Episode length cap, in **environment** steps (so `1000 * frame_skip` physics steps). |

---

## Observation space — `Box(9,)`, unbounded

| Index | Name | Description |
|---|---|---|
| 0 | `sin(pitch)` | Encodes pitch angle without the wraparound discontinuity a raw angle has |
| 1 | `cos(pitch)` | Paired with index 0; together they give the policy an unambiguous angle |
| 2 | `pitch_rate` | Angular velocity about body Y, rad/s |
| 3 | `yaw_rate` | Angular velocity about body Z, rad/s |
| 4 | `forward_speed` | Linear velocity along body X (forward), m/s |
| 5 | `lateral_speed` | Linear velocity along body Y (sideways), m/s |
| 6 | `left_wheel_speed` | Left wheel joint angular velocity, rad/s |
| 7 | `right_wheel_speed` | Right wheel joint angular velocity, rad/s |
| 8 | `height` | Base link Z position (world frame), m |

**Design notes:**
- Body-frame linear velocity (indices 4–5) is computed by rotating the world-frame free-joint
  velocity into the base's own frame (`rotation.T @ linear_velocity`), so "forward speed" always
  means forward *relative to the robot*, regardless of which way it's facing.
- `sin`/`cos` of pitch is used instead of raw pitch specifically so the observation space has
  no discontinuity at ±180°. With `TIP_THRESHOLD` at 40°, this mainly future-proofs the encoding
  rather than solving a problem this task currently hits.
- Height (index 8) doubles as a fall detector in termination (below) and gives the policy direct
  feedback on how close it is to that boundary.
- No target velocity/heading is included — this observation is **balance-only**. Extending to a
  driving task means adding a target/command to this vector (and the reward).

---

## Action space — `Box(2,)`, range `[-1, 1]`

`[left_action, right_action]` → scaled by `max_torque = 0.5` before being written to
`data.ctrl[left_motor_id]` / `data.ctrl[right_motor_id]`. So the physical torque range is
**±0.5 N·m** per wheel, independent of whatever `ctrlrange` is set in the XML — the env does its
own clipping and scaling rather than reading the actuator's limits.

> **Check this against your XML's `ctrlrange`.** If the actuator's real limit is smaller than
> 0.5, the env will command torques the actuator can't produce (MuJoCo will silently clip at the
> actuator level); if it's larger, you're leaving torque headroom unused.

---

## Reward — computed every `env.step()` (i.e. once per `frame_skip` physics steps)

```
reward =   1.0                                            # alive bonus
         - 3.0    * pitch²
         - 0.05   * pitch_rate²
         - 0.05   * yaw_rate²                              # yaw_rate_penalty
         - 0.02   * forward_speed²
         - 0.01   * (left_wheel_speed² + right_wheel_speed²)  # wheel_speed_penalty
         - 0.001  * (left_action² + right_action²)          # action cost
         - 0.01   * (left_action - right_action)²           # differential_torque_penalty
         - 10.0                                             # only if terminated this step
```

| Term | Weight | Purpose |
|---|---|---|
| Alive bonus | `+1.0` | Baseline reward for surviving each step; makes longer episodes worth more |
| Pitch penalty | `3.0` | Dominant term — the core "stay upright" signal |
| Pitch-rate penalty | `0.05` | Discourages violent recovery motions, encourages settling smoothly |
| Yaw-rate penalty | `0.05` | Discourages spinning in place as a side effect of balancing |
| Forward-speed penalty | `0.02` | Discourages drifting off, similar in spirit to the PID script's outer velocity loop |
| Wheel-speed penalty | `0.01` | Discourages "solving" balance by spinning wheels fast with little net effect |
| Action-magnitude penalty | `0.001` | Small effort penalty; encourages smoother, lower-torque control |
| Differential-torque penalty | `0.01` | Discourages the two wheels fighting each other / unwanted turning |
| Fall penalty | `-10.0` | One-time penalty applied on the terminating step, on top of losing all future alive bonus |

All weights are set as instance attributes in `__init__` except the pitch, pitch-rate, forward-speed,
and action weights, which are hardcoded in `step()` — so to tune those you currently edit the
`step()` method directly rather than passing a constructor argument. (`max_torque`,
`wheel_speed_penalty`, `yaw_rate_penalty`, and `differential_torque_penalty` are the only ones set
as `self.` attributes, and only `max_torque` is exposed as a constructor argument's sibling; none
are actual `__init__` parameters yet — all are fixed defaults.)

---

## Episode termination

| Condition | Trigger | Type |
|---|---|---|
| Tipped over | `abs(pitch) > 40°` | `terminated = True` |
| Fell / collapsed | `height < 0.025 m` | `terminated = True` |
| Time limit | `step_count >= 1000` | `truncated = True` |

Two independent ways to detect a "fall" are used here — pitch angle and base height — which
catches failure modes a pitch-only check could miss (e.g. the robot collapsing straight down
without necessarily exceeding the pitch threshold, depending on the model's geometry).

---

## Reset

- `mujoco.mj_resetData()` puts the model back to its XML-defined initial state.
- A random pitch is drawn uniformly from **±0.08 rad (≈ ±4.6°)** and applied as a quaternion
  rotation about the body's local Y axis, composed with whatever orientation the XML defines
  at rest.
- No initial velocity (linear or angular) is randomized — every episode starts at rest, only the
  lean angle varies.
- `step_count` resets to 0.

> The quaternion multiply in `_reset_state` is a correct Hamilton product — it looks like it's
> missing terms compared to the general formula, but that's because the tilt quaternion's `x` and
> `z` components are exactly 0 (it's a pure rotation about Y), so those terms drop out on their
> own. It's correct for any starting orientation, not just an upright one.

---

## Rendering

- `render_mode="human"` opens a passive MuJoCo viewer on first `render()` call and syncs it every
  step; `render_mode=None` runs headless (what you want for parallelized training).
- `close()` shuts the viewer down if one was opened.

---

## Quick-reference: what to tune, and where

| Want to change... | Edit... |
|---|---|
| How hard it's penalized for leaning | `3.0` pitch weight in `step()` |
| Max torque available | `self.max_torque` in `__init__` |
| Episode length | `self.max_steps` in `__init__` |
| Fall detection sensitivity | `40°` / `0.025 m` thresholds in `step()` |
| Initial tilt randomization range | `±0.08` in `_reset_state()` |
| Action smoothness / effort | `0.001` action-cost weight in `step()` |
| Physics steps per action | `frame_skip` constructor argument |

---

## Setup

All commands below assume `projects/balancia_robot/` as the working directory:

```bash
cd projects/balancia_robot
pip install -r requirements.txt
# requirements: gymnasium, mujoco, numpy, stable-baselines3
# For TensorBoard logging also ensure tensorboard is installed:
pip install "stable-baselines3[extra]" tensorboard
```

> The training script at `scripts/train_rl.py:21` does `sys.path.insert(0, PROJECT_ROOT)` so `envs.balancing_robot_env` is importable regardless of cwd, but running from the project root keeps log/model paths consistent.

---

## Training

The PPO training entry point is `scripts/train_rl.py:40` (`TwoWheeledBalanceEnv` with `Monitor` + `make_vec_env`).

| Script | Class used | Notes |
|---|---|---|
| `scripts/train_rl.py:35` | `TwoWheeledBalanceEnv(render_mode=None)` | Vectorized with `n_envs` parallel envs; headless for speed |

### 1. Train from scratch (default 1M timesteps, 8 parallel envs)

```bash
python scripts/train_rl.py
# equivalent to:
python scripts/train_rl.py --timesteps 1000000 --n-envs 8
```

Hyperparameters set in `scripts/train_rl.py:60`:
`MlpPolicy`, `n_steps=2048`, `batch_size=256`, `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `ent_coef=0.0`.

### 2. Longer run

```bash
python scripts/train_rl.py --timesteps 2000000
```

### 3. Resume from last checkpoint

```bash
python scripts/train_rl.py --resume
# or resume + extend:
python scripts/train_rl.py --resume --timesteps 500000
```

`--resume` at `scripts/train_rl.py:56` loads `trained_models/ppo_balance_latest.zip` and continues with `reset_num_timesteps=False`.

### 4. Adjust parallelism

```bash
python scripts/train_rl.py --n-envs 4   # lower if CPU-bound
```

### Outputs

| Path | What |
|---|---|
| `trained_models/ppo_balance_latest.zip` | Final model (overwritten each run) — saved at `scripts/train_rl.py:86` |
| `trained_models/checkpoints/ppo_balance_<steps>_steps.zip` | Periodic checkpoint every `50_000 // n_envs` env steps via `CheckpointCallback` at `scripts/train_rl.py:73` (e.g. `ppo_balance_500000_steps.zip`) |
| `logs/ppo_balance_1/events.out.tfevents.*` | TensorBoard logs (`tensorboard_log=logs`, `tb_log_name="ppo_balance"` at `scripts/train_rl.py:68`) |

> Docstring in `scripts/train_rl.py:5` still refers to `train_ppo.py` — the actual file is `scripts/train_rl.py`.

---

## Playing / Watching the Policy

Load a trained PPO checkpoint and render with the passive MuJoCo viewer (`scripts/play_rl.py:45` uses `TwoWheeledBalanceEnv(render_mode="human")`).

```bash
# Watch the latest model (default)
python scripts/play_rl.py

# Watch a specific checkpoint
python scripts/play_rl.py --model trained_models/checkpoints/ppo_balance_500000_steps.zip

# Watch the final model with a forward velocity command
python scripts/play_rl.py --model trained_models/ppo_balance_latest.zip --target-vel 0.3
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `trained_models/ppo_balance_latest.zip` | Path to `.zip` saved by SB3 — see `scripts/play_rl.py:34` |
| `--target-vel` | `0.0` | Constant `env.target_vel` (m/s) set at `scripts/play_rl.py:48`; effective only if the env's velocity tracking is enabled |

Runtime overlay prints `pitch`, `forward_velocity`, and `reward` each step (`scripts/play_rl.py:57`). The viewer runs in real-time (`env.dt` sleep at `scripts/play_rl.py:69`), resets automatically on `terminated`/`truncated`, and exits cleanly on `Ctrl+C` / window close.

**Headless check (no viewer):**

```python
from envs.balancing_robot_env import TwoWheeledBalanceEnv
from stable_baselines3 import PPO

model = PPO.load("trained_models/ppo_balance_latest.zip")
env = TwoWheeledBalanceEnv(render_mode=None)
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

---

## Visualizing Data

### TensorBoard — training curves

```bash
tensorboard --logdir logs
# then open http://localhost:6006
# to scope to one run:
tensorboard --logdir logs/ppo_balance_1
```

Logged by SB3 at `scripts/train_rl.py:58`/`71` (`tensorboard_log=str(LOG_DIR)`). Key scalars: `rollout/ep_rew_mean`, `rollout/ep_len_mean`, `train/*` losses, `time/fps`.

If `tensorboard` is not found, `pip install tensorboard`.

### Checkpoints — compare training stages

```bash
ls -lh trained_models/checkpoints/
# watch an early vs late policy:
python scripts/play_rl.py --model trained_models/checkpoints/ppo_balance_100000_steps.zip
python scripts/play_rl.py --model trained_models/checkpoints/ppo_balance_1000000_steps.zip
```

### Model viewer — inspect the MuJoCo model without a policy

```bash
python test/view_model.py
# loads assets/two_wheeled.xml at test/view_model.py:5 and calls mujoco.viewer.launch
```

### PID baseline — compare classical control vs RL

```bash
python test/pid_balance.py
# Interactive: click viewer window, then UP/DOWN (speed), LEFT/RIGHT (turn), ENTER (stop)
# Gains defined at test/pid_balance.py:31 — KP=9.0, KD=0.5, KV=0.15, KT=0.3
```

Prints `pitch`, `v/v_cmd`, `yaw_rate`, and `torque` each physics step (`test/pid_balance.py:256`). Useful to sanity-check the model before training.

### Quick evaluation snippet — collect episode stats without rendering

```python
from envs.balancing_robot_env import TwoWheeledBalanceEnv
from stable_baselines3 import PPO
import numpy as np

model = PPO.load("trained_models/ppo_balance_latest.zip")
env = TwoWheeledBalanceEnv()

for ep in range(5):
    obs, _ = env.reset()
    rews = []
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, terminated, truncated, info = env.step(action)
        rews.append(rew)
        if terminated or truncated:
            break
    print(f"ep {ep}: len={len(rews)} return={sum(rews):.1f} last_pitch={info['pitch']:.3f}")
```

Plot `rews` or `info` history with `matplotlib` for custom analysis.