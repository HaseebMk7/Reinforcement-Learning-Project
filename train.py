from stable_baselines3 import PPO
from tb4_env import TurtleBot4Env
import os
import shutil

# 1. CLEANUP (Force fresh start)
if os.path.exists("ppo_turtlebot4_maze.zip"):
    os.remove("ppo_turtlebot4_maze.zip")
    print("Deleted old model.")

# 2. Setup
env = TurtleBot4Env(render_mode=None)

# 3. Train
print("Initializing PPO... (Using CPU)")
model = PPO("MlpPolicy", env, verbose=1, device='cpu', learning_rate=0.0003)

print("Starting Training...")
# This config learns VERY fast. 
model.learn(total_timesteps=250000)

model.save("ppo_turtlebot4_maze")
print("DONE. Run enjoy.py now.")
