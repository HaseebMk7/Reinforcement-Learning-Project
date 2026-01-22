import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import mujoco
import mujoco.viewer
import os
import glob
import numpy as np

class RobotBridge(Node):
    def __init__(self):
        super().__init__('mujoco_bridge')
        # Subscribe to standard cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.listener_callback,
            10)
        
        # Robot Parameters (Approximate for TurtleBot4)
        self.wheel_radius = 0.036  # meters
        self.wheel_separation = 0.235 # meters
        
        # Current desired wheel velocities
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0
        
        print("ROS 2 Node Started: Listening on /cmd_vel")

    def listener_callback(self, msg):
        # Convert Linear/Angular to Left/Right Wheel Speeds
        linear = msg.linear.x
        angular = msg.angular.z
        
        # Differential Drive Kinematics
        # v_left = v - (w * d / 2)
        # v_right = v + (w * d / 2)
        v_l = linear - (angular * self.wheel_separation / 2.0)
        v_r = linear + (angular * self.wheel_separation / 2.0)
        
        # Convert from linear speed (m/s) to angular speed (rad/s) for the motors
        # This assumes the MuJoCo actuators are velocity servos.
        # If they are torque motors, this might need tuning (or higher values).
        scale_factor = 1.0 / self.wheel_radius 
        
        self.left_wheel_vel = v_l * scale_factor
        self.right_wheel_vel = v_r * scale_factor

def main():
    # 1. SETUP ROS
    rclpy.init()
    bridge = RobotBridge()

    # 2. SETUP MUJOCO
    # (Same file finding logic as before)
    print("Searching for model...")
    search_pattern = os.path.join(os.getcwd(), "ai-enhanced-ros", "**", "turtlebot4.xml")
    found_files = glob.glob(search_pattern, recursive=True)
    if not found_files:
        print("Error: Model not found!")
        return
    
    xml_path = found_files[0]
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 3. RUN SIMULATION LOOP
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # A. Process ROS messages (non-blocking)
            rclpy.spin_once(bridge, timeout_sec=0)
            
            # B. Apply controls to MuJoCo
            # Assuming Actuator 0 is Left, Actuator 1 is Right (Standard)
            # We multiply by a gain if the robot is too sluggish
            gain = 1.0 
            data.ctrl[0] = bridge.left_wheel_vel * gain
            data.ctrl[1] = bridge.right_wheel_vel * gain

            # C. Step Physics
            mujoco.mj_step(model, data)
            viewer.sync()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
