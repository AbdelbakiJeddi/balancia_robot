from pathlib import Path
import math
import os
import signal
import threading
import time

import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = str(
    Path(__file__).resolve().parents[1]
    / "assets"
    / "two_wheeled.xml"
)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ============================================================
# PARAMETERS
# ============================================================

# --- inner loop: pitch PID ---
KP = 9.0
KI = 0.0
KD = 0.5
CONTROL_SIGN = +1.0     # flip if it falls faster with the controller on

# --- outer loop: speed -> desired lean ---
KV = 0.15
MAX_LEAN = math.radians(8.0)
VEL_SIGN = +1.0         # flip if it accelerates away from the speed target

# --- yaw (turning) loop ---
KT = 0.3                # differential torque per rad/s of yaw-rate error
TURN_SIGN = +1.0        # flip if LEFT arrow turns the robot right (or it spins away)
MAX_TURN_TORQUE = 2.0

# --- keyboard driving ---
SPEED_STEP = 0.05       # m/s added per UP/DOWN key press
MAX_SPEED = 0.5         # m/s
TURN_STEP = 0.5         # rad/s added per LEFT/RIGHT key press
MAX_TURN = 2.0          # rad/s
ACCEL_LIMIT = 0.5       # m/s^2   how fast the speed command ramps
TURN_ACCEL_LIMIT = 4.0  # rad/s^2 how fast the turn command ramps

INTEGRAL_LIMIT = 0.5
TIP_THRESHOLD = math.radians(40.0)
INITIAL_TILT_DEG = 5.0


# ============================================================
# IDs
# ============================================================

left_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor")
right_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

assert left_motor_id >= 0, "left_motor not found in model"
assert right_motor_id >= 0, "right_motor not found in model"
assert base_id >= 0, "base_link not found in model"

base_jnt_id = model.body_jntadr[base_id]
assert model.jnt_type[base_jnt_id] == mujoco.mjtJoint.mjJNT_FREE, \
    "base_link needs a free joint"
base_qpos_adr = model.jnt_qposadr[base_jnt_id]
base_dof_adr = model.jnt_dofadr[base_jnt_id]


# ============================================================
# KEYBOARD COMMANDS
# ============================================================

# GLFW key codes
KEY_ENTER = 257
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265

# Targets set by the keyboard (the control loop ramps toward these)
command = {"speed": 0.0, "turn": 0.0}


def key_callback(keycode):
    """Each press nudges the target. Enter stops everything."""
    if keycode == KEY_UP:
        command["speed"] += SPEED_STEP
    elif keycode == KEY_DOWN:
        command["speed"] -= SPEED_STEP
    elif keycode == KEY_LEFT:
        command["turn"] += TURN_STEP      # +yaw = counter-clockwise = turn left
    elif keycode == KEY_RIGHT:
        command["turn"] -= TURN_STEP
    elif keycode == KEY_ENTER:
        command["speed"] = 0.0
        command["turn"] = 0.0

    command["speed"] = float(np.clip(command["speed"], -MAX_SPEED, MAX_SPEED))
    command["turn"] = float(np.clip(command["turn"], -MAX_TURN, MAX_TURN))


# ============================================================
# CLEAN CTRL+C
# ============================================================

running = True


def handle_sigint(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, handle_sigint)


# ============================================================
# HELPERS
# ============================================================

def get_pitch():
    R = data.xmat[base_id].reshape(3, 3)
    return math.atan2(-R[2, 0], R[0, 0])


def get_pitch_rate():
    # Free-joint angular velocity is body-frame: +4 is about local Y
    return data.qvel[base_dof_adr + 4]


def get_yaw_rate():
    # Body-frame angular velocity about local Z
    return data.qvel[base_dof_adr + 5]


def get_forward_speed():
    """Speed along the robot's forward (+X local) direction, in m/s."""
    R = data.xmat[base_id].reshape(3, 3)
    v_world = data.qvel[base_dof_adr : base_dof_adr + 3]   # world-frame linear velocity
    return float(np.dot(R[:, 0], v_world))


def clamp_to_ctrlrange(value, actuator_id):
    if model.actuator_ctrllimited[actuator_id]:
        lo, hi = model.actuator_ctrlrange[actuator_id]
        return float(np.clip(value, lo, hi))
    return value


def ramp(current, target, max_step):
    return current + float(np.clip(target - current, -max_step, max_step))


# ============================================================
# RESET + INITIAL TILT
# ============================================================

mujoco.mj_resetData(model, data)

theta = math.radians(INITIAL_TILT_DEG)
w1, x1, y1, z1 = math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0
w2, x2, y2, z2 = data.qpos[base_qpos_adr + 3 : base_qpos_adr + 7]

data.qpos[base_qpos_adr + 3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
data.qpos[base_qpos_adr + 4] = w1*x2 + x1*w2 + y1*z2 - z1*y2
data.qpos[base_qpos_adr + 5] = w1*y2 - x1*z2 + y1*w2 + z1*x2
data.qpos[base_qpos_adr + 6] = w1*z2 + x1*y2 - y1*x2 + z1*w2

mujoco.mj_forward(model, data)


print("======================================")
print(" Two-Wheeled Robot - Balance + Drive")
print("======================================")
print("Click the viewer window, then:")
print("  UP / DOWN     : faster forward / backward")
print("  LEFT / RIGHT  : turn left / right")
print("  ENTER         : stop (speed and turn to 0)")
print("  Ctrl+C        : quit")
print()


# ============================================================
# SIMULATION
# ============================================================

dt = model.opt.timestep
integral = 0.0
tipped = False
v_cmd = 0.0      # smoothed speed command actually used by the controller
turn_cmd = 0.0   # smoothed yaw-rate command

try:
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.05]
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -25

        while running and viewer.is_running():
            step_start = time.time()

            # ---- ramp commands toward keyboard targets ----
            v_cmd = ramp(v_cmd, command["speed"], ACCEL_LIMIT * dt)
            turn_cmd = ramp(turn_cmd, command["turn"], TURN_ACCEL_LIMIT * dt)

            # ---- state ----
            pitch = get_pitch()
            pitch_rate = get_pitch_rate()
            yaw_rate = get_yaw_rate()
            v_fwd = get_forward_speed()

            if abs(pitch) > TIP_THRESHOLD:
                tipped = True

            if tipped:
                torque = 0.0
                turn_torque = 0.0
                pitch_ref = 0.0
            else:
                # Outer loop: speed error -> desired lean.
                # Moving faster than commanded -> lean back; slower -> lean forward.
                pitch_ref = -VEL_SIGN * KV * (v_fwd - v_cmd)
                pitch_ref = float(np.clip(pitch_ref, -MAX_LEAN, MAX_LEAN))

                # Inner loop: PID on pitch error
                error = pitch - pitch_ref
                integral += error * dt
                integral = float(np.clip(integral, -INTEGRAL_LIMIT, INTEGRAL_LIMIT))
                torque = CONTROL_SIGN * (KP * error + KI * integral + KD * pitch_rate)

                # Yaw loop: differential torque tracks the commanded turn rate
                # (with turn_cmd = 0 it also holds the robot driving straight)
                turn_torque = TURN_SIGN * KT * (turn_cmd - yaw_rate)
                turn_torque = float(np.clip(turn_torque, -MAX_TURN_TORQUE, MAX_TURN_TORQUE))

            # Left/right wheels get the shared balance torque +/- the turn torque
            data.ctrl[left_motor_id] = clamp_to_ctrlrange(torque - turn_torque, left_motor_id)
            data.ctrl[right_motor_id] = clamp_to_ctrlrange(torque + turn_torque, right_motor_id)

            mujoco.mj_step(model, data)
            viewer.sync()

            print(
                f"\r"
                f"pitch={math.degrees(pitch):+6.2f} deg | "
                f"v={v_fwd:+5.2f}/{v_cmd:+5.2f} m/s | "
                f"yaw={yaw_rate:+5.2f}/{turn_cmd:+5.2f} rad/s | "
                f"torque={torque:+6.3f}"
                f"{'  [TIPPED]' if tipped else ''}   ",
                end=""
            )

            remaining = dt - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        print("\nClosing viewer...")
        viewer.close()

finally:
    print("\nViewer closed safely.")
    # Last resort: if a viewer thread blocks exit, force it after 2 s
    threading.Timer(2.0, lambda: os._exit(0)).start()