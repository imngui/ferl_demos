import numpy as np
import time
import math

from rclpy.impl import rcutils_logger
logger = rcutils_logger.RcutilsLogger(name="pid")


class NPID(object):
    def __init__(self, num_joints):
        # Initialize arrays to store the gains and constraints for all joints
        self.num_joints = num_joints
        self.k_p = np.zeros(num_joints).reshape((self.num_joints, 1))
        self.k_i = np.zeros(num_joints).reshape((self.num_joints, 1))
        self.k_d = np.zeros(num_joints).reshape((self.num_joints, 1))
        self.i_clamp = np.zeros(num_joints).reshape((self.num_joints, 1))
        self.k_ff = np.zeros(num_joints).reshape((self.num_joints, 1))
        # self.max_velocity = np.full(num_joints, np.inf)  # Default to no velocity limit

        # Initialize arrays for storing the integral term and previous errors
        self.integral_term = np.zeros(num_joints).reshape((self.num_joints, 1))
        self.previous_position = np.zeros(num_joints).reshape((self.num_joints, 1))
        self._last_time = None
        self.previous_position = np.zeros((num_joints, 1))
        self.i = 0

    @property
    def last_time(self):
       """ Read-only access to the last time. """
       return self._last_time

    def update_gains(self, k_p, k_i, k_d, i_clamp, k_ff):
        # Update the gains for all joints
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.i_clamp = i_clamp
        self.k_ff = k_ff

    # def set_max_velocity(self, max_velocity):
    #     # Set the maximum velocity constraints for all joints
    #     self.max_velocity = max_velocity

    def calculate_control(self, measured_position, desired_position, measured_velocity, desired_velocity=None, dt=None, is_continuous=False):
        # Calculate the position error
        if dt == None:
            cur_time = time.time()
            if self._last_time is None:
                self._last_time = cur_time 
            dt = cur_time - self._last_time
            self._last_time = cur_time

        # dp_str = np.array2string(desired_position.reshape((1,-1)))
        # logger.info(f'des pos: {dp_str}')
        # mp_str = np.array2string(measured_position.reshape((1,-1)))
        # logger.info(f'mes pos: {mp_str}')
        
        position_error = desired_position - measured_position
        # pe_str = np.array2string(position_error.reshape((1,-1)))
        # logger.info(f'pos err: {pe_str}')

        if dt == 0 or math.isnan(dt) or math.isinf(dt):
            return np.zeros(len(self.k_p)).reshape((self.num_joints, 1))
        if is_continuous:
            position_error = np.arctan2(np.sin(position_error), np.cos(position_error))

        if desired_velocity is None:
            desired_velocity = position_error / dt

        # Calculate the velocity error
        # velocity_error = desired_velocity - measured_velocity
        velocity_error = (measured_position - self.previous_position) / dt
        # ve_str = np.array2string(velocity_error.reshape((1,-1)))
        # logger.info(f'vel err: {ve_str}\n')

        # Proportional term
        P_term = self.k_p * position_error

        # Integral term
        self.integral_term += self.k_i * position_error * dt
        # Apply integral clamp
        self.integral_term = np.clip(self.integral_term, -self.i_clamp, self.i_clamp)

        # Derivative term (based on velocity error)
        D_term = self.k_d * velocity_error

        # Feedforward term (based on desired velocity)
        F_f = self.k_ff * desired_velocity

        # Control output (velocity or effort command)
        control_output = F_f + P_term + self.integral_term + D_term

        # Store the current error as previous error for the next iteration
        self.previous_error = position_error

        self.previous_position = measured_position

        # self.i += 1
        # if self.i >= 50:
        #     assert(1 == 2)

        return control_output