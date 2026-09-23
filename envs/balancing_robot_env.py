"""
Gymnasium environment for the two-wheeled balancing robot - SYMMETRIC DRIVE.

Observation (7,):
    [pitch, pitch_rate, yaw_rate, forward_velocity,
     left_wheel_velocity, right_wheel_velocity, target_velocity]
    # left/right wheel vels are equal by construction (symmetric torque)

Action (1,):
    [torque] in [-1, 1], scaled and applied IDENTICALLY to left+right motors
    # System is symmetric: no differential torque, no turning. Both wheels always same velocity.

Reward, per step:
     + ALIVE_BONUS
     - PITCH_WEIGHT        * pitch^2
     - PITCH_RATE_WEIGHT   * pitch_rate^2
     - YAW_RATE_WEIGHT     * yaw_rate^2  (should stay ~0 with symmetric drive)
     - VEL_WEIGHT          * (forward_velocity - target_velocity)^2
     - WHEEL_SPEED_WEIGHT  * (left_wheel_vel^2 + right_wheel_vel^2)
     - ACTION_WEIGHT       * action^2    (single scalar)
     - ACTION_RATE_WEIGHT  * (action - prev_action)^2

target_velocity is no longer a single random value snapped on at reset. It is
ramped smoothly from 0 toward a goal (CHANGE LOG item 2), and the goal is
resampled a few times per episode when randomize_target_vel=True (item 3),
so the policy learns to track a moving command rather than reach one speed
and hold it.

Episode ends (terminated) if the robot tips past TIP_THRESHOLD.
Episode ends (truncated) after MAX_EPISODE_STEPS.
"""

from pathlib import Path
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "assets" / "two_wheeled.xml"
)

TIP_THRESHOLD = math.radians(35.0)     # a bit more lenient than the PID script's tip cutoff
MAX_EPISODE_STEPS = 5000

ALIVE_BONUS = 1.0
PITCH_WEIGHT = 7.0
PITCH_RATE_WEIGHT = 0.05
YAW_RATE_WEIGHT = 0.02
VEL_WEIGHT = 0.4            # CHANGED: was 3.0, too strong relative to pitch
WHEEL_SPEED_WEIGHT = 0.001
ACTION_WEIGHT = 0.001
ACTION_RATE_WEIGHT = 0.05   # NEW: penalizes spiking/chattering between steps

# Randomization at reset
INIT_TILT_RANGE_DEG = 13.0       # uniform in [-range, +range]
INIT_TILT_RATE_RANGE = 0.5       # rad/s, small push at start

# Velocity command
MAX_TARGET_VEL = 0.5             # CHANGED: was 3.0 -- verify this against what your
                                  # PID drive script can actually sustain, then adjust.
VEL_RAMP_RATE = 0.3              # m/s per second, how fast target_vel chases its goal
VEL_RESAMPLE_STEPS = 300         # resample the goal this often (steps), 0 = disable


class TwoWheeledBalanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, randomize_target_vel=False):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self.render_mode = render_mode
        self.randomize_target_vel = randomize_target_vel
        self._viewer = None

        # ---- ids ----
        self.left_motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor"
        )
        self.right_motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor"
        )
        self.base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        assert self.left_motor_id >= 0, "left_motor not found in model"
        assert self.right_motor_id >= 0, "right_motor not found in model"
        assert self.base_id >= 0, "base_link not found in model"

        base_jnt_id = self.model.body_jntadr[self.base_id]
        assert self.model.jnt_type[base_jnt_id] == mujoco.mjtJoint.mjJNT_FREE, \
            "base_link needs a free joint"
        self.base_qpos_adr = self.model.jnt_qposadr[base_jnt_id]
        self.base_dof_adr = self.model.jnt_dofadr[base_jnt_id]

        # ---- wheel joints, for observing wheel speed (optional but useful) ----
        self.left_wheel_dof = self._wheel_dof("left_wheel_joint")
        self.right_wheel_dof = self._wheel_dof("right_wheel_joint")

        # ---- torque limits, to scale actions from [-1, 1] ----
        self.left_torque_limit = self._torque_limit(self.left_motor_id)
        self.right_torque_limit = self._torque_limit(self.right_motor_id)

        # ---- spaces ----
        # Symmetric: one torque command drives BOTH wheels identically
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        obs_high = np.array([np.inf] * 7, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        self.target_vel = 0.0
        self._target_vel_goal = 0.0
        self._prev_action = np.zeros(1, dtype=np.float32)
        self._step_count = 0

    # ------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------
    def _wheel_dof(self, joint_name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return None  # model may name wheel joints differently; obs will use 0.0
        return self.model.jnt_dofadr[jid]

    def _torque_limit(self, actuator_id):
        if self.model.actuator_ctrllimited[actuator_id]:
            lo, hi = self.model.actuator_ctrlrange[actuator_id]
            return float(max(abs(lo), abs(hi)))
        return 5.0  # fallback if the actuator has no ctrlrange set

    # ------------------------------------------------------------
    # state readers (same conventions as your PID scripts)
    # ------------------------------------------------------------
    def _get_pitch(self):
        R = self.data.xmat[self.base_id].reshape(3, 3)
        return math.atan2(-R[2, 0], R[0, 0])

    def _get_pitch_rate(self):
        return self.data.qvel[self.base_dof_adr + 4]

    def _get_yaw_rate(self):
        return self.data.qvel[self.base_dof_adr + 5]

    def _get_forward_speed(self):
        R = self.data.xmat[self.base_id].reshape(3, 3)
        v_world = self.data.qvel[self.base_dof_adr : self.base_dof_adr + 3]
        return float(np.dot(R[:, 0], v_world))

    def _get_wheel_speeds(self):
        lv = self.data.qvel[self.left_wheel_dof] if self.left_wheel_dof is not None else 0.0
        rv = self.data.qvel[self.right_wheel_dof] if self.right_wheel_dof is not None else 0.0
        return float(lv), float(rv)

    def _get_obs(self):
        pitch = self._get_pitch()
        pitch_rate = self._get_pitch_rate()
        yaw_rate = self._get_yaw_rate()
        v_fwd = self._get_forward_speed()
        lv, rv = self._get_wheel_speeds()
        return np.array(
            [pitch, pitch_rate, yaw_rate, v_fwd, lv, rv, self.target_vel],
            dtype=np.float32,
        )

    # ------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------
    def reset(self, *, seed=None, randomize=True, **kwargs):
        super().reset(seed=seed)
        rng = self.np_random

        mujoco.mj_resetData(self.model, self.data)

        # Random small initial tilt so the policy learns to recover, not just
        # one fixed fall direction.
        tilt_deg = rng.uniform(-INIT_TILT_RANGE_DEG, INIT_TILT_RANGE_DEG) if randomize else 3.0
        theta = math.radians(tilt_deg)
        w1, x1, y1, z1 = math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0
        w2, x2, y2, z2 = self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7]

        self.data.qpos[self.base_qpos_adr + 3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        self.data.qpos[self.base_qpos_adr + 4] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        self.data.qpos[self.base_qpos_adr + 5] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        self.data.qpos[self.base_qpos_adr + 6] = w1*z2 + x1*y2 - y1*x2 + z1*w2

        if randomize:
            self.data.qvel[self.base_dof_adr + 4] = rng.uniform(
                -INIT_TILT_RATE_RANGE, INIT_TILT_RATE_RANGE
            )

        mujoco.mj_forward(self.model, self.data)

        # CHANGED: target_vel always starts at 0 (matches the robot's actual resting
        # state) and ramps toward a goal instead of snapping to it at t=0.
        self.target_vel = 0.0
        self._target_vel_goal = (
            float(rng.uniform(-MAX_TARGET_VEL, MAX_TARGET_VEL))
            if self.randomize_target_vel
            else 0.0
        )
        self._prev_action = np.zeros(1, dtype=np.float32)
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # Backward compat: if an old 2D action is passed, average it to a symmetric scalar
        if action.size == 2:
            action = np.array([float(np.mean(action))], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # NEW: resample the velocity goal periodically, so the policy learns to
        # track a changing command rather than reach one speed and hold it.
        if (
            self.randomize_target_vel
            and VEL_RESAMPLE_STEPS > 0
            and self._step_count > 0
            and self._step_count % VEL_RESAMPLE_STEPS == 0
        ):
            self._target_vel_goal = float(
                self.np_random.uniform(-MAX_TARGET_VEL, MAX_TARGET_VEL)
            )

        # NEW: ramp target_vel toward the goal instead of a step change
        ramp_step = VEL_RAMP_RATE * self.dt
        self.target_vel += float(
            np.clip(self._target_vel_goal - self.target_vel, -ramp_step, ramp_step)
        )

        torque = float(action[0])
        # Symmetric drive: same torque to both wheels
        left_torque = torque * self.left_torque_limit
        right_torque = torque * self.right_torque_limit

        self.data.ctrl[self.left_motor_id] = left_torque
        self.data.ctrl[self.right_motor_id] = right_torque

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        pitch = self._get_pitch()
        pitch_rate = self._get_pitch_rate()
        yaw_rate = self._get_yaw_rate()
        v_fwd = self._get_forward_speed()
        lv, rv = self._get_wheel_speeds()

        terminated = abs(pitch) > TIP_THRESHOLD
        truncated = self._step_count >= MAX_EPISODE_STEPS

        vel_error = v_fwd - self.target_vel
        action_delta = action - self._prev_action
        self._prev_action = action.copy()

        pitch_pen = PITCH_WEIGHT * pitch ** 2
        pitch_rate_pen = PITCH_RATE_WEIGHT * pitch_rate ** 2
        yaw_pen = YAW_RATE_WEIGHT * yaw_rate ** 2
        vel_pen = VEL_WEIGHT * vel_error ** 2
        wheel_pen = WHEEL_SPEED_WEIGHT * (lv ** 2 + rv ** 2)
        action_pen = ACTION_WEIGHT * float(action[0] ** 2)
        action_rate_pen = ACTION_RATE_WEIGHT * float(np.sum(action_delta ** 2))

        reward = (
            ALIVE_BONUS
            - pitch_pen
            - pitch_rate_pen
            - yaw_pen
            - vel_pen
            - wheel_pen
            - action_pen
            - action_rate_pen
        )
        if terminated:
            reward -= 10.0  # extra penalty for falling, on top of losing future alive bonus

        obs = self._get_obs()
        wheel_avg = 0.5 * (lv + rv)
        info = {
            "pitch": pitch,
            "pitch_rate": pitch_rate,
            "yaw_rate": yaw_rate,
            "forward_velocity": v_fwd,
            "target_velocity": self.target_vel,
            "target_velocity_goal": self._target_vel_goal,
            "wheel_speeds": (lv, rv),
            "wheel_avg": wheel_avg,
            "reward_breakdown": {
                "alive": ALIVE_BONUS,
                "pitch_pen": pitch_pen,
                "pitch_rate_pen": pitch_rate_pen,
                "yaw_pen": yaw_pen,
                "vel_pen": vel_pen,
                "wheel_pen": wheel_pen,
                "action_pen": action_pen,
                "action_rate_pen": action_rate_pen,
            },
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        import mujoco.viewer as mjv
        if self._viewer is None:
            self._viewer = mjv.launch_passive(self.model, self.data)
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._viewer.cam.lookat[:] = [0.0, 0.0, 0.05]
            self._viewer.cam.distance = 0.8
            self._viewer.cam.azimuth = 45
            self._viewer.cam.elevation = -25
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# Optional: register so you can also do gym.make("TwoWheeledBalance-v0")
try:
    gym.register(
        id="TwoWheeledBalance-v0",
        entry_point="envs.two_wheeled_env:TwoWheeledBalanceEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
    )
except gym.error.Error:
    pass  # already registered (e.g. module reloaded)