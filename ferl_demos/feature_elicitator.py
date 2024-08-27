#! /usr/bin/env python3

from ament_index_python.packages import get_package_share_directory

import torch

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor # Enables the description of parameters

import math
import sys, select, os
import time

# srv imports
from controller_manager_msgs.srv import SwitchController
# from kortex_hardware.srv import ModeService

# msg imports
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# other defined imports
from ferl.controllers.pid_controller import PIDController
from ferl.planners.trajopt_planner import TrajoptPlanner
from ferl.learners.phri_learner import PHRILearner
from ferl.utils import ros2_utils, openrave_utils
from ferl.utils.environment import Environment
from ferl.utils.trajectory import Trajectory

import numpy as np
import pickle
import ast
import threading


def convert_string_array_to_dict(string_array):
    feat_range_dict = {}
    for item in string_array:
        key, value = item.split(':')
        feat_range_dict[key] = float(value)  # Convert the value to a float
    return feat_range_dict
    
def convert_string_array_to_dict_of_lists(string_array):
    object_centers_dict = {}
    for item in string_array:
        key, value = item.split(':')
        # Use ast.literal_eval to safely evaluate the string as a Python list
        object_centers_dict[key] = ast.literal_eval(value)
    return object_centers_dict
    
def get_parameter_as_dict(string_array):
    """
    Convert a StringArray parameter to a dictionary with the appropriate Python types.
    """
    converted_dict = {}
    for item in string_array:
        key, value_str = item.split(':', 1)  # Split on the first colon
        converted_dict[key] = convert_string_to_appropriate_type(value_str)
    return converted_dict

def convert_string_to_appropriate_type(value_str):
    """
    Attempt to convert a string to its appropriate Python type.
    """
    try:
        # Try to evaluate the string as a Python literal (e.g., list, dict, int, float, bool, None)
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # If evaluation fails, return the string as is
        return value_str

class FeatureElicitator(Node):

    def __init__(self):
        super().__init__('feature_elicitator')

        # Load parameters.
        self.load_params()

        # Setup subscribers and publishers.
        self.register_callbacks()

        # Run the main loop.
        self.run_thread = threading.Thread(target=self.run)
        self.run_thread.start()
        # self.run()

    def load_params(self):
        """
		Loading parameters and setting up variables from the ROS environment.
		"""
        # Declare parameters
        self.declare_parameter('setup.prefix', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.model_filename', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.object_centers', descriptor=ParameterDescriptor(dynamic_typing=True))  # Declaring object_centers as a map
        self.declare_parameter('setup.feat_list', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.feat_weights', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.start', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.goal', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.goal_pose', None, descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.T', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.timestep', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.save_dir', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.INTERACTION_TORQUE_THRESHOLD', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.INTERACTION_TORQUE_EPSILON', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.FEAT_RANGE', descriptor=ParameterDescriptor(dynamic_typing=True))  # Declaring FEAT_RANGE as a map
        self.declare_parameter('setup.LF_dict', descriptor=ParameterDescriptor(dynamic_typing=True))  # Declaring LF_dict as a map
        self.declare_parameter('setup.CONFIDENCE_THRESHOLD', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.N_QUERIES', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.nb_layers', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('setup.nb_units', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('planner.type', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('planner.max_iter', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('planner.num_waypts', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.type', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.p_gain', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.i_gain', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.d_gain', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.epsilon', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('controller.max_cmd', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('learner.type', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('learner.step_size', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('learner.alpha', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('learner.n', descriptor=ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('learner.P_beta', descriptor=ParameterDescriptor(dynamic_typing=True))  # Declaring P_beta as a map


        # ----- General Setup ----- #
        self.prefix = self.get_parameter('setup.prefix').value
        pick = self.get_parameter('setup.start').value
        self.start = np.array(pick)#*(math.pi/180.0)
        place = self.get_parameter('setup.goal').value
        self.goal = np.array(place)#*(math.pi/180.0)
        self.goal_pose = self.get_parameter('setup.goal_pose').value
        self.T = self.get_parameter('setup.T').value
        self.timestep = self.get_parameter('setup.timestep').value
        self.save_dir = self.get_parameter('setup.save_dir').value
        self.INTERACTION_TORQUE_THRESHOLD = self.get_parameter('setup.INTERACTION_TORQUE_THRESHOLD').value
        self.INTERACTION_TORQUE_EPSILON = self.get_parameter('setup.INTERACTION_TORQUE_EPSILON').value
        self.CONFIDENCE_THRESHOLD = self.get_parameter('setup.CONFIDENCE_THRESHOLD').value
        self.N_QUERIES = self.get_parameter('setup.N_QUERIES').value
        self.nb_layers = self.get_parameter('setup.nb_layers').value
        self.nb_units = self.get_parameter('setup.nb_units').value

        # Openrave parameters for the environment.
        model_filename = self.get_parameter('setup.model_filename').value
        object_centers = get_parameter_as_dict(self.get_parameter('setup.object_centers').value)
        # print("object_centers: ", object_centers)
        feat_list = self.get_parameter('setup.feat_list').value
        weights = self.get_parameter('setup.feat_weights').value
        FEAT_RANGE = get_parameter_as_dict(self.get_parameter('setup.FEAT_RANGE').value)
        # print("FEAT_RANGE: ", FEAT_RANGE)
        feat_range = [FEAT_RANGE[feat_list[feat]] for feat in range(len(feat_list))]
        LF_dict = get_parameter_as_dict(self.get_parameter('setup.LF_dict').value)
        self.environment = Environment(model_filename, object_centers, feat_list, feat_range, np.array(weights), LF_dict)
        self.num_dofs = self.environment.env.GetRobots()[0].GetActiveDOF()
        
        # Get joint names
        self.joint_names = [self.environment.env.GetRobots()[0].GetJointFromDOFIndex(i).GetName() for i in self.environment.env.GetRobots()[0].GetManipulator('arm').GetArmIndices()]
        print("num_joints: ", len(self.joint_names))
        print("joint_names: ", self.joint_names)


        # ----- Planner Setup ----- #
        # Retrieve the planner specific parameters.
        planner_type = self.get_parameter('planner.type').value
        if planner_type == "trajopt":
            max_iter = self.get_parameter('planner.max_iter').value
            num_waypts = self.get_parameter('planner.num_waypts').value

            # Initialize planner and compute trajectory to track.
            self.planner = TrajoptPlanner(max_iter, num_waypts, self.environment)
            # TODO: Implement TrajoptPlanner class.
        else:
            raise Exception('Planner {} not implemented.'.format(planner_type))
        
        # time.sleep(30)
        
        print("Planning trajectory...")
        print("Start: ", self.start)
        print("Goal: ", self.goal)
        print("Goal Pose: ", self.goal_pose)
        tt = time.time()
        self.traj = self.planner.replan(self.start, self.goal, self.goal_pose, self.T, self.timestep)
        self.get_logger().info(f"Planning time: {time.time() - tt}")
        self.traj_plan = self.traj.downsample(self.planner.num_waypts)

        # Track if you have reached the start/goal of the path.
        self.reached_start = False
        self.reached_goal = False
        self.feature_learning_mode = False
        self.interaction_mode = False

        # Save the intermediate target configuration.
        self.curr_pos = None

        # Track data and keep stored.
        self.interaction_data = []
        self.interaction_time = []
        self.feature_data = []
        self.track_data = False


        # ----- Controller Setup ----- #
        # Retrieve controller specific parameters.
        controller_type = self.get_parameter('controller.type').value
        if controller_type == "pid":
            # P, I, D gains.
            # TODO: Change np.eye(7) to correct arm dofs.
            P = self.get_parameter('controller.p_gain').value * np.eye(self.num_dofs)
            I = self.get_parameter('controller.i_gain').value * np.eye(self.num_dofs)
            D = self.get_parameter('controller.d_gain').value * np.eye(self.num_dofs)

            # Stores proximity threshold.
            epsilon = self.get_parameter('controller.epsilon').value

            # Stores maximum COMMANDED joint torques.
            MAX_CMD = self.get_parameter('controller.max_cmd').value

            self.controller = PIDController(P, I, D, epsilon, MAX_CMD)
            # TODO: Implement PIDController class.
        else:
            raise Exception('Controller {} not implemented.'.format(controller_type))
        
        # Planner tells controller what plan to follow.
        self.controller.set_trajectory(self.traj)

        # Stores the current COMMANDED joint torques.
        # TODO: Change np.eye(7) to correct arm dofs.
        self.cmd = np.eye(self.num_dofs)


        # ----- Learner Setup ----- #
        # Retrieve learner specific parameters.
        constants = {}
        constants["step_size"] = self.get_parameter('learner.step_size').value
        constants["P_beta"] = get_parameter_as_dict(self.get_parameter('learner.P_beta').value)
        constants["alpha"] = self.get_parameter('learner.alpha').value
        constants["n"] = self.get_parameter('learner.n').value
        self.feat_method = self.get_parameter('learner.type').value
        self.learner = PHRILearner(self.feat_method, self.environment, constants)
        # TODO: Implement PHRILearner class.
        self.get_logger().info("Finished Init")

    def register_callbacks(self):
        """
        Set up all the subscribers and publishers needed.
        """
        # TODO: Figure out how to define the correct messages.
        # Create joint-velocity publisher.
        self.vel_pub = self.create_publisher(JointTrajectory, self.prefix + '/in/joint_velocity', 1)

        # Create subscriber to joint_angles.
        self.joint_angles_sub = self.create_subscription(JointState, '/joint_states', self.joint_angles_callback, 1)
        # Create subscriber to joint_torques.
        self.joint_torques_sub = self.create_subscription(JointState, self.prefix + '/out/joint_torques', self.joint_torques_callback, 1)

    def joint_angles_callback(self, msg):
        """
        Reads the latest position of the robot and publishes an
        appropriate torque command to move the robot to the target.
        """

        # Read the current joint angles from the robot.
        # TODO: Find a more generic way to do this for different dof robots.
        curr_pos = np.array(msg.position).reshape(self.num_dofs,1)
        # curr_pos = np.array([msg.joint1, msg.joint2, msg.joint3, msg.joint4, msg.joint5, msg.joint6, msg.joint7]).reshape(7, 1)

        # Convert to radians.
        curr_pos = curr_pos#*(math.pi/180.0)

        # Check if we are in feature learning mode.
        if self.feature_learning_mode:
            # Allow the person to mvoe the end effector with no control resistance.
            self.cmd = np.zeros((self.num_dofs, self.num_dofs))

            # If we are tracking feature data, update raw features and time.
            if self.track_data:
                # Add recording to feature data.
                self.feature_data.append(self.environment.raw_features(curr_pos))
            return
        
        # When no in feature learning stage, update position.
        self.curr_pos = curr_pos

        # Update cmd from PID based on current position.
        self.cmd = self.controller.get_command(self.curr_pos)

        # Check if start/goal has been reached.
        if self.controller.path_start_T is not None:
            self.reached_start = True
        if self.controller.path_end_T is not None:
            self.reached_goal = True
    
    def joint_torques_callback(self, msg):
        """
        Reads the latest torque sensed by the robot and records it for
        plotting & analysis.
        """

        # Read the current joint torques from the robot.
        torque_curr = np.array(msg.effort).reshape(self.num_dofs,1)
        # torque_curr = np.array([msg.joint1, msg.joint2, msg.joint3, msg.joint4, msg.joint5, msg.joint6, msg.joint7]).reshape(7, 1)
        interaction = False
        for i in range(self.num_dofs):
            # Center torques around zero.
            torque_curr[i][0] -= self.INTERACTION_TORQUE_THRESHOLD[i]
            # Check if interaction was not noise.
            if np.fabs(torque_curr[i][0]) > self.INTERACTION_TORQUE_EPSILON[i] and self.reached_start:
                interaction = True
        
        if interaction:
            if self.reached_start and not self.reached_goal:
                timestamp = time.time() - self.controller.path_start_T
                self.interaction_data.append(torque_curr)
                self.interaction_time.append(timestamp)
                if self.interaction_mode == False:
                    self.interaction_mode = True
        else:
            if self.interaction_mode:
                # Check if betas are above CONFIDENCE_THRESHOLD.
                betas = self.learner.learn_betas(self.traj, self.interaction_data[0], self.interaction_time[0])
                if max(betas) < self.CONFIDENCE_THRESHOLD:
                    # We must learn a new feature that passes CONFIDENCE_THRESHOLD before resuming.
                    print("The robot does not understand the input!")
                    self.feature_learning_mode = True
                    feature_learning_timestamp = time.time()
                    input_size = len(self.environment.raw_features(torque_curr))
                    self.environment.new_learned_feature(self.nb_layers, self.nb_units)
                    while True:
                        # Keep asking for input until we are confident.
                        for i in range(self.N_QUERIES):
                            print("Need more data to learn the feature!")
                            self.feature_data = []

                            # Request the person to place the robot in a low feature value state.
                            print("Place the robot in a low feature value state and press ENTER when ready.")
                            line = sys.stdin.readline()
                            self.track_data = True

                            print("Place the robot in a high feature value state and press ENTER when ready.")
                            line = sys.stdin.readline()
                            self.track_data = False

                            # Pre-process the recorded data before training.
                            feature_data = np.squeeze(np.array(self.feature_data))
                            lo = 0
                            hi = feature_data.shape[0] - 1
                            while np.linalg.norm(feature_data[lo] - feature_data[lo + 1]) < 0.01 and lo < hi:
                                lo += 1
                            while np.linalg.norm(feature_data[hi] - feature_data[hi - 1]) < 0.01 and hi > 0:
                                hi -= 1
                            feature_data = feature_data[lo:hi + 1, :]
                            print("Collected {} samples out of {}.".format(feature_data.shape[0], len(self.feature_data)))

                            # Provide optional start and end labels.
                            start_label = 0.0
                            end_label = 1.0
                            print("Would you like to label your start? Press ENTER if not or enter a number from 0-10")
                            line = sys.stdin.readline()
                            if line in [str(i) for i in range(11)]:
                                start_label = int(i) / 10.0

                            print("Would you like to label your goal? Press ENTER if not or enter a number from 0-10")
                            line = sys.stdin.readline()
                            if line in [str(i) for i in range(11)]:
                                end_label = int(i) / 10.0

                            # Add the newly collected data.
                            self.environment.learned_features[-1].add_data(feature_data, start_label, end_label)
                        
                        # Train new feature with data of increasing "goodness".
                        self.environment.learned_features[-1].train()

                        # Check if we are happy with the input.
                        print("Are you happy with the training? (y/n)")
                        line = sys.stdin.readline()
                        if line == "yes" or line == "Y" or line == "y":
                            break
                    
                    # Compute new beta for the new feature.
                    beta_new = self.learner.learn_betas(self.traj, torque_curr, timestamp, [self.environment.num_features - 1])[0]
                    betas.append(beta_new)

                    # Move time forward to return to interaction position.
                    self.controller.path_start_T += (time.time() - feature_learning_timestamp)

                # We do no have misspecification now, so resume reward learning.
                self.feature_learning_mode = False
                
                # learn reward.
                for i in range(len(self.interaction_data)):
                    self.learner.learn_weights(self.traj, self.interaction_data[i], self.interaction_time[i], betas)
                self.traj = self.planner.replan(self.start, self.goal, self.goal_pose, self.T, self.timestep, seed=self.traj_plan.waypts)
                self.traj_plan = self.traj.downsample(self.planner.num_waypts)
                self.controller.set_trajectory(self.traj)

                # Turn off interaction mode.
                self.interaction_mode = False
                self.interaction_data = []
                self.interaction_time = []
                            

    def run(self):
        """
        Main loop for the feature elicitation process.
        """

        # Start admittance control mode.
        # TODO: Implement ros2_utils
        # ros2_utils.start_admittance_mode(self.prefix, self)

        # Publish to ROS at 100hz.
        rate = self.create_rate(1)

        print("----------------------------------")
        print("Moving robot, type Q to quit:")

        while rclpy.ok():
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if line[0] == 'q' or line == 'Q' or line == 'quit':
                    break

            # TODO: Implement ros2_utils
            print("HERE")
            self.vel_pub.publish(ros2_utils.cmd_to_JointTrajMsg(self.joint_names, self.cmd))
            rate.sleep()

        self.get_logger().info("----------------------------------")
        # TODO: Implement ros2_utils
        # ros2_utils.stop_admittance_mode(self.prefix, node)

def main(args=None):
    rclpy.init(args=args)

    feature_elicitator = FeatureElicitator()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(feature_elicitator)

    try:
        executor.spin()
    finally:
        feature_elicitator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
    

