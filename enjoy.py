import mujoco
import mujoco.viewer
import time
from stable_baselines3 import PPO
from tb4_env import TurtleBot4Env

print("Loading...")
env = TurtleBot4Env(render_mode=None)

try:
    model = PPO.load("ppo_turtlebot4_maze", device='cpu')
except:
    print("Error: Model not found. Did you finish training?")
    exit()

obs, _ = env.reset()
print("Running Demo...")

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    viewer.sync()
    while viewer.is_running():
        # Deterministic=True usually works best for navigation
        # If it freezes, toggle this to False
        action, _states = model.predict(obs, deterministic=True)
        
        obs, rewards, dones, truncated, info = env.step(action)
        
        viewer.sync()
        time.sleep(0.01)
        
        if dones or truncated:
            obs, _ = env.reset()
