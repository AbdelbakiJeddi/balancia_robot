# Balancia Robot

Two-wheeled balancing robot trained with PPO (Stable-Baselines3) in MuJoCo.

## What's in this repo

- MuJoCo model of a two-wheeled robot (`assets/two_wheeled.xml`)
- Gymnasium environment with symmetric wheel drive (`envs/balancing_robot_env.py`)
- PPO training script (`scripts/train_rl.py`)
- Script to watch a trained policy in the viewer (`scripts/play_rl.py`)
- A classical PID controller for comparison (`test/pid_balance.py`)
- Basic sanity-check scripts (`test/view_model.py`, `test/test_model.py`)

**Observation (7 values):** pitch, pitch rate, yaw rate, forward velocity, left wheel velocity, right wheel velocity, target velocity

**Action:** single torque value, applied symmetrically to both wheels

**Reward:** staying alive, penalized for pitch error, velocity tracking error, and jerky actions; large penalty for falling

**Episode ends when:** the robot tilts past 35°, or 5000 steps are reached

See `docs/ENV_REFERENCE.md` for the full observation/action/reward tables.

## Setup

### 1. Install MuJoCo

```bash
pip install mujoco
```

(MuJoCo's Python bindings include the simulator itself — no separate install needed.)

### 2. Clone the repo

```bash
git clone https://github.com/<you>/balancia-robot.git
cd balancia-robot
```

### 3. Create an environment and install the rest of the requirements

```bash
python -m venv env_mujoco
source env_mujoco/bin/activate
pip install -r requirements.txt
```

This installs `gymnasium`, `mujoco`, `numpy`, `stable-baselines3`, and `tensorboard`.

## Try it out

Check that the MuJoCo model loads and opens a viewer:

```bash
python test/view_model.py
```

Run the classical PID controller as a baseline (use ↑/↓ in the viewer to change target speed):

```bash
python test/pid_balance.py
```

## Train

```bash
python scripts/train_rl.py --timesteps 1000000 --n-envs 8
```

Watch training progress:

```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006`.

To resume a previous run:

```bash
python scripts/train_rl.py --resume --timesteps 500000
```

## Watch a trained policy

```bash
python scripts/play_rl.py
```

Or point it at a specific checkpoint:

```bash
python scripts/play_rl.py --model trained_models/checkpoints/ppo_balance_2015808_steps.zip --target-vel 0.3
```

**Note:** the most recent checkpoint isn't always the best one — training can overfit. In one run, `ppo_balance_2015808_steps.zip` balanced reliably (0/20 falls over 20 test seeds) while the final `ppo_balance_latest.zip` fell more often. Worth comparing a few checkpoints rather than assuming later is better.
