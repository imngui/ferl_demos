import rclpy
# import kinova_msgs_ros2.msg
import geometry_msgs.msg
# import kinova_msgs_ros2.srv
# from kinova_msgs_ros2.srv import *
from moveit_msgs.msg import CartesianTrajectory, CartesianTrajectoryPoint
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController

import time

def cmd_to_JointTorqueMsg(cmd):
	"""
	Returns a JointTorque Kinova msg from an array of torques
	"""
	jointCmd = JointTrajectoryPoint()
	jointCmd.effort[0] = cmd[0][0]
	jointCmd.effort[1] = cmd[1][1]
	jointCmd.effort[2] = cmd[2][2]
	jointCmd.effort[3] = cmd[3][3]
	jointCmd.effort[4] = cmd[4][4]
	jointCmd.effort[5] = cmd[5][5]
	jointCmd.effort[6] = cmd[6][6]
	
	return jointCmd

def cmd_to_JointTrajMsg(joint_names, cmd):
	"""
	Returns a 
	"""
	joint_cmd = JointTrajectoryPoint()
	joint_vel = [cmd[i][i] for i in range(len(joint_names))]
	joint_cmd.velocities = joint_vel
	# jointCmd.velocities[0] = cmd[0][0]
	# jointCmd.velocities[1] = cmd[1][1]
	# jointCmd.velocities[2] = cmd[2][2]
	# jointCmd.velocities[3] = cmd[3][3]
	# jointCmd.velocities[4] = cmd[4][4]
	# jointCmd.velocities[5] = cmd[5][5]
	# jointCmd.velocities[6] = cmd[6][6]
	joint_cmd.time_from_start = rclpy.duration.Duration(seconds=1.0).to_msg()
 
	traj_msg = JointTrajectory()
	traj_msg.joint_names = joint_names
 
	traj_msg.points = [joint_cmd]
 
	return traj_msg

def waypts_to_PoseArrayMsg(cart_waypts):
	"""
	Returns a PoseArray msg from an array of 3D carteian waypoints
	"""
	poseArray = geometry_msgs.msg.PoseArray()
	# poseArray = C
	# TODO: figure out what to use instead of rospy.Time.now()
	# poseArray.header.stamp = rospy.Time.now()
	poseArray.header.stamp = time.time()
	poseArray.header.frame_id = "/root"

	for i in range(len(cart_waypts)):
		somePose = geometry_msgs.msg.Pose()
		somePose.position.x = cart_waypts[i][0]
		somePose.position.y = cart_waypts[i][1]
		somePose.position.z = cart_waypts[i][2]

		somePose.orientation.x = 0.0
		somePose.orientation.y = 0.0
		somePose.orientation.z = 0.0
		somePose.orientation.w = 1.0
		poseArray.poses.append(somePose)

	return poseArray

def start_admittance_mode(prefix, node):
	"""
	Switches Kinova arm to admittance-control mode using ROS services.
	"""
	node.get_logger().info("Starting gen3 compliance controller")
	cs_cli = node.create_service(SwitchController, '/controller_manager/switch_controller')
	while not cs_cli.wait_for_service(timeout_sec=1.0):
		node.get_logger().info('service not available, waiting again...')
	req = SwitchController.Request()
	req.start_controllers = ['joint_space_compliant_controller']
	req.stop_controllers = ['velocity_controller', 'task_space_compliant_controller']
	req.strictness = 1
 
	future = cs_cli.call_async(req)
	rclpy.spin_until_future_complete(node, future)
 
	if future.result() is not None:
		node.get_logger().info(f'Success: {future.result().ok}')
	else:
		node.get_logger().error('Exception while calling service: {0}'.format(future.exception()))
 

def stop_admittance_mode(prefix, node):
	"""
	Switches Kinova arm to position-control mode using ROS services.
	"""
	node.get_logger().info("Stopping gen3 compliance controller")
	cs_cli = node.create_service(SwitchController, '/controller_manager/switch_controller')
	while not cs_cli.wait_for_service(timeout_sec=1.0):
		node.get_logger().info('service not available, waiting again...')
	req = SwitchController.Request()
	req.start_controllers = ['velocity_controller']
	req.stop_controllers = ['joint_space_compliant_controller', 'task_space_compliant_controller']
	req.strictness = 1

	future = cs_cli.call_async(req)
	rclpy.spin_until_future_complete(node, future)
 
	if future.result() is not None:
		node.get_logger().info(f'Success: {future.result().ok}')
	else:
		node.get_logger().error('Exception while calling service: {0}'.format(future.exception()))
