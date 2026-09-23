import time
import math
from pathlib import Path
import mujoco
import mujoco.viewer

# ... load model, set KP/KD, get IDs, etc. ...

tipped = False

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # camera setup
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0, 0, 0.05]
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -25

        print("Simulation running. Close the window or press Ctrl+C to stop.")

        while viewer.is_running():
            step_start = time.time()

            pitch = get_pitch()
            pitch_rate = data.cvel[base_id][4]

            if abs(pitch) > TIP_THRESHOLD:
                tipped = True

            if tipped:
                torque = 0.0
            else:
                torque = KP * pitch + KD * pitch_rate   # or -torque if sign is wrong

            data.ctrl[left_motor_id] = torque
            data.ctrl[right_motor_id] = torque

            mujoco.mj_step(model, data)
            viewer.sync()

            # optional status print
            print(f"\rpitch={math.degrees(pitch):+7.2f}°  torque={torque:+6.3f}", end="")

            # real-time
            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

except KeyboardInterrupt:
    print("\nStopped by Ctrl+C")
finally:
    print("\nViewer closed safely.")