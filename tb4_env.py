import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
import glob
import shutil

class TurtleBot4Env(gym.Env):
    def __init__(self, render_mode=None):
        super(TurtleBot4Env, self).__init__()
        
        # --- 1. ASSET FIXER ---
        cwd = os.getcwd()
        xml_search = os.path.join(cwd, "**", "turtlebot4.xml")
        found_xmls = glob.glob(xml_search, recursive=True)
        if not found_xmls:
            raise FileNotFoundError("Could not find turtlebot4.xml!")
        robot_xml_path = os.path.abspath(found_xmls[0])
        robot_dir = os.path.dirname(robot_xml_path)

        extensions = ["*.stl", "*.obj", "*.mtl", "*.png", "*.jpg"]
        for ext in extensions:
            search_pattern = os.path.join(cwd, "**", ext)
            found_files = glob.glob(search_pattern, recursive=True)
            for file_path in found_files:
                dest_path = os.path.join(robot_dir, os.path.basename(file_path))
                if not os.path.exists(dest_path):
                    try: shutil.copy2(file_path, dest_path)
                    except: pass 

        # --- 2. DEFINE OBSTACLES (x, y, radius_for_math) ---
        # We define them here so the Python math knows where they are
        self.obstacles = [
            {'x': 1.5, 'y': 1.5, 'size': 0.4},  # Red Box 1
            {'x': -1.5, 'y': -1.5, 'size': 0.4} # Red Box 2
        ]

        # --- 3. GENERATE MAZE WITH OBSTACLES ---
        maze_xml_content = f"""
        <mujoco model="turtlebot_maze">
          <include file="{robot_xml_path}"/>
          <visual><headlight ambient="0.4 0.4 0.4"/></visual>
          <asset>
             <texture name="grid_tex" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".8 .8 .8"/>
             <material name="grid" texture="grid_tex" texrepeat="1 1" texuniform="true" reflectance=".2"/>
          </asset>
          <worldbody>
            <geom name="floor_arena" size="5 5 0.1" type="plane" material="grid" condim="3"/>
            
            <geom name="wall_n" pos="0 2.6 0" size="2.6 0.1 0.5" type="box" rgba="0.5 0.5 0.5 1"/>
            <geom name="wall_s" pos="0 -2.6 0" size="2.6 0.1 0.5" type="box" rgba="0.5 0.5 0.5 1"/>
            <geom name="wall_e" pos="2.6 0 0" size="0.1 2.5 0.5" type="box" rgba="0.5 0.5 0.5 1"/>
            <geom name="wall_w" pos="-2.6 0 0" size="0.1 2.5 0.5" type="box" rgba="0.5 0.5 0.5 1"/>
            
            <geom name="obs1" pos="1.5 1.5 0" size="0.4 0.4 0.5" type="box" rgba="0.8 0.2 0.2 1"/>
            
            <geom name="obs2" pos="-1.5 -1.5 0" size="0.4 0.4 0.5" type="box" rgba="0.8 0.2 0.2 1"/>

            <body name="target_mocap" mocap="true" pos="0 0 0">
                <geom type="sphere" size="0.3" rgba="0 1 0 1" conaffinity="0" contype="0"/>
            </body>
          </worldbody>
        </mujoco>
        """
        
        self.generated_xml = "auto_maze.xml"
        with open(self.generated_xml, "w") as f:
            f.write(maze_xml_content)
            
        self.model = mujoco.MjModel.from_xml_path(self.generated_xml)
        self.data = mujoco.MjData(self.model)
        
        self.viewer = None
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Action: [Linear Vel, Angular Vel]
        self.action_space = spaces.Box(low=np.array([-0.3, -2.0]), 
                                       high=np.array([0.5, 2.0]), dtype=np.float32)
        
        # Observation (7 Inputs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        self.target = np.array([0.0, 0.0]) 
        self.max_steps = 2500 
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # --- SMART SPAWNING LOGIC ---
        valid_pos = False
        while not valid_pos:
            # 1. Pick a random spot
            candidate = np.random.uniform(-2.0, 2.0, size=2)
            valid_pos = True
            
            # 2. Check if it's inside any Obstacle (with 0.6m margin)
            for obs in self.obstacles:
                # Simple box collision check
                dx = abs(candidate[0] - obs['x'])
                dy = abs(candidate[1] - obs['y'])
                # If target is inside the box (plus some padding), try again
                if dx < (obs['size'] + 0.3) and dy < (obs['size'] + 0.3):
                    valid_pos = False
            
            # 3. Check if it's too close to robot start (0,0)
            if np.linalg.norm(candidate) < 0.5:
                valid_pos = False

        self.target = candidate
        
        try:
            mocap_id = self.model.body("target_mocap").mocapid[0]
            self.data.mocap_pos[mocap_id][0:2] = self.target
            self.data.mocap_pos[mocap_id][2] = 0.2
        except: pass
        
        self.current_step = 0
        if self.viewer: self.viewer.sync()
        return self._get_obs(), {}

    def step(self, action):
        v, w = action
        self.data.ctrl[0] = (v - w) * 10.0
        self.data.ctrl[1] = (v + w) * 10.0
        
        for _ in range(5): mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        if self.viewer: self.viewer.sync()
        
        obs = self._get_obs()
        dist_to_goal = obs[0]
        
        reward = -dist_to_goal 
        reward -= 0.05 
        
        terminated = False
        truncated = False
        
        if dist_to_goal < 0.5:
            reward += 500
            terminated = True
            print(f"Goal Reached!")

        # Check collisions (Walls OR Obstacles)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # If we hit ANYTHING static (wall or obs), punish.
            # Usually static objects are geom2 with id > robot parts
            reward_penalty = 0
            
            # Simple check: If robot hits something, is it a wall or obs?
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            hit_list = ["wall", "obs"]
            if any(x in str(name1) for x in hit_list) or any(x in str(name2) for x in hit_list):
                 reward -= 50
                 terminated = True
                 print("Crashed into Obstacle/Wall!")
                 break

        if self.current_step >= self.max_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # 1. Robot State
        x, y = self.data.qpos[0], self.data.qpos[1]
        qw, qx, qy, qz = self.data.qpos[3:7]
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        
        dx = self.target[0] - x
        dy = self.target[1] - y
        dist_to_goal = np.sqrt(dx**2 + dy**2)
        angle_to_target = np.arctan2(dy, dx) - yaw
        
        # --- ADVANCED VIRTUAL LIDAR (Ray Tracing) ---
        # Default distances (to the outer walls)
        dist_n = 2.5 - y
        dist_s = y - (-2.5)
        dist_e = 2.5 - x
        dist_w = x - (-2.5)
        
        # Now check if an Obstacle blocks the view!
        for obs in self.obstacles:
            size = obs['size'] + 0.2 # Add padding for robot width
            
            # Check NORTH Beam
            # Is the obstacle horizontally aligned with us?
            if abs(obs['x'] - x) < size:
                # Is the obstacle North of us?
                if obs['y'] > y:
                    # Is it closer than the wall?
                    dist_to_obs = (obs['y'] - size) - y
                    if dist_to_obs < dist_n: dist_n = dist_to_obs

            # Check SOUTH Beam
            if abs(obs['x'] - x) < size:
                if obs['y'] < y:
                    dist_to_obs = y - (obs['y'] + size)
                    if dist_to_obs < dist_s: dist_s = dist_to_obs

            # Check EAST Beam
            if abs(obs['y'] - y) < size:
                if obs['x'] > x:
                    dist_to_obs = (obs['x'] - size) - x
                    if dist_to_obs < dist_e: dist_e = dist_to_obs

            # Check WEST Beam
            if abs(obs['y'] - y) < size:
                if obs['x'] < x:
                    dist_to_obs = x - (obs['x'] + size)
                    if dist_to_obs < dist_w: dist_w = dist_to_obs
        
        # Clamp values so they don't go negative (if inside an object)
        dist_n = max(0.0, dist_n)
        dist_s = max(0.0, dist_s)
        dist_e = max(0.0, dist_e)
        dist_w = max(0.0, dist_w)
        
        return np.array([
            dist_to_goal,
            np.sin(angle_to_target), 
            np.cos(angle_to_target), 
            dist_n,
            dist_s,
            dist_e,
            dist_w
        ], dtype=np.float32)
