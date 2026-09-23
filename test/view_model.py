from pathlib import Path
import mujoco
import mujoco.viewer

MODEL_PATH = str(Path(__file__).resolve().parents[1] / "assets" / "two_wheeled.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Launches the fully interactive viewer (blocks until window is closed)
mujoco.viewer.launch(model, data)