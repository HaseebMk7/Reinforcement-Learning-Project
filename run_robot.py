import mujoco
import mujoco.viewer
import time
import os
import glob

# --- SMART FILE FINDER ---
# Look for turtlebot4.xml anywhere inside the current folder
print("Searching for turtlebot4.xml...")
search_pattern = os.path.join(os.getcwd(), "ai-enhanced-ros", "**", "turtlebot4.xml")
found_files = glob.glob(search_pattern, recursive=True)

if not found_files:
    # Fallback: look for ANY xml if specific one is missing
    print("Could not find 'turtlebot4.xml'. Listing ALL .xml files found:")
    all_xmls = glob.glob(os.path.join(os.getcwd(), "ai-enhanced-ros", "**", "*.xml"), recursive=True)
    for f in all_xmls:
        print(f" - {f}")
    print("ERROR: Please update the script to use one of the files above.")
    exit()

xml_path = found_files[0]
print(f"SUCCESS: Found model at: {xml_path}")
# -------------------------

print("Loading model...")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation Running. Robot should be moving!")
    
    # Loop forever
    while viewer.is_running():
        # Apply force to wheels (actuators 0 and 1)
        data.ctrl[0] = 5.0 
        data.ctrl[1] = 5.0 

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)
