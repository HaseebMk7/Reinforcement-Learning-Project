# TurtleBot 4 Reinforcement Learning Navigation (Sim-to-Real)

This project implements a Deep Reinforcement Learning (DRL) agent capable of controlling a TurtleBot 4 to navigate point-to-point while avoiding obstacles. The system is designed with a **Sim-to-Real workflow**: train in a fast, physics-based simulation (MuJoCo), then deploy the trained model onto the physical robot using ROS 2 Humble.

---

##  Project Structure

```
├── assets/               # 3D models (XML, STL) for the simulation
├── deploy/               # Scripts for running on the real robot
│   └── run_real_robot.py # The ROS 2 "Bridge" node
├── envs/                 # Custom Gymnasium Environment
│   └── tb4_env.py        # MuJoCo physics & reward logic
├── models/               # Saved trained agents (.zip files)
├── data/                 # Training logs and tensorboard data
├── train.py              # Main training script (PPO algorithm)
├── enjoy.py              # Test/visualize trained models
├── ros_bridge.py         # ROS 2 integration utilities
└── requirements.txt      # Python dependencies
```

---

##  Key Features

- **Algorithm**: Proximal Policy Optimization (PPO) via `stable-baselines3`
- **Simulation**: High-fidelity physics using MuJoCo with native ray-casting to simulate Lidar
- **Observation Space (5 Inputs)**:
  - Lidar Distances (Left, Front, Right)
  - Distance to Target
  - Heading Angle to Target
- **Sim-to-Real Bridge**: A specialized ROS 2 node that downsamples real 360° Lidar scans into the sparse 3-ray format the agent expects
- **Smart Spawning**: Custom logic ensures the robot, target, and dynamic obstacles never spawn overlapping each other

---

##  Installation

### 1. Requirements

Ensure you have **Python 3.8+** and **ROS 2 Humble** (for deployment).

```bash
pip install gymnasium stable-baselines3 shimmy>=1.0.0 mujoco
```

### 2. Assets

Ensure `turtlebot4.xml` and associated mesh files are placed in the `assets/` folder. The environment will automatically load them.

---

##  Training (Simulation)

To train the agent from scratch in the MuJoCo environment:

```bash
python train.py
```

### What happens:

- The robot attempts to reach the **Green Sphere (Target)**
- **Reward Function**:
  - `+100` for reaching the goal
  - `+ (2.0 - dist)` dense reward for moving closer
  - `-50` for collisions (Walls or Red Obstacles)
  - `-0.05` per step (time penalty to encourage speed)
- The model is saved periodically to the `models/` directory

---

##  Deployment (Real Robot)

Once the agent achieves a high reward in simulation, you can deploy it to the physical TurtleBot 4.

### Prerequisites:

- The TurtleBot 4 must be running and connected to the network
- Your PC must have ROS 2 Humble installed and configured to communicate with the robot

### Run the Agent:

```bash
# Make sure you source ROS 2 first
source /opt/ros/humble/setup.bash

# Run the bridge script
python3 deploy/run_real_robot.py
```

### How it works:

1. The script subscribes to `/scan` (Lidar) and `/odom` (Position)
2. It downsamples the massive Lidar array into just 3 rays (Left, Front, Right) to match the simulation
3. It feeds this data to the trained PPO model
4. It publishes velocity commands to `/cmd_vel` to drive the robot

---

## Environment Details

### State Space (Observations)

| Index | Name         | Range       | Description                        |
|-------|--------------|-------------|------------------------------------|
| 0     | Lidar Left   | [0, 3.5]    | Distance to obstacle at +90°       |
| 1     | Lidar Front  | [0, 3.5]    | Distance to obstacle at 0°         |
| 2     | Lidar Right  | [0, 3.5]    | Distance to obstacle at -90°       |
| 3     | Target Dist  | [0, Inf]    | Euclidean distance to goal         |
| 4     | Target Angle | [-π, π]     | Relative heading to goal           |

### Action Space (Continuous)

| Index | Name        | Range         | Description                     |
|-------|-------------|---------------|---------------------------------|
| 0     | Linear Vel  | [-0.2, 0.4]   | Forward/Backward speed (m/s)    |
| 1     | Angular Vel | [-1.5, 1.5]   | Turning speed (rad/s)           |

---

##  Troubleshooting

### Reward stuck at -103

This usually means the agent is "freezing" to avoid penalties. We fixed this by:
- Adding a **Dense Reward** (giving points for every inch moved closer to the target)
- Reducing the collision penalty relative to the success bonus

### ImportError: No module named 'rospy'

You are running ROS 2. Ensure you use `rclpy` (included in `deploy/run_real_robot.py`) and not the ROS 1 `rospy` library.

### Robot spins in circles on real hardware

Check your Lidar indices in `run_real_robot.py`. Ensure "Front" is actually index 0 (or correct for your specific Lidar model).

---

## Monitoring Training

You can monitor training progress using TensorBoard:

```bash
tensorboard --logdir=./data/
```

Or use the provided plotting scripts:

```bash
python plot_results.py
```

---

##  Testing Trained Models

To visualize a trained model in simulation:

```bash
python enjoy.py
```

This will load the best model and render the agent's behavior in the MuJoCo viewer.

---

## License

This project is open-source. Please refer to individual component licenses for MuJoCo, stable-baselines3, and ROS 2.

---

##  Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [TurtleBot 4 Documentation](https://turtlebot.github.io/turtlebot4-user-manual/)
- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)
