""" 
A simple PID controller class.  

This is a mostly literal C++ -> Python translation of the ROS
control_toolbox Pid class: http://ros.org/wiki/control_toolbox.
"""

import time
import math
import numpy as np

from rclpy.impl import rcutils_logger
logger = rcutils_logger.RcutilsLogger(name="pid")

#*******************************************************************
# Translated from pid.cpp by Nathan Sprague
# Jan. 2013 (Modified Jan. 2014)
# See below for original license information:
#*******************************************************************

#******************************************************************* 
# Software License Agreement (BSD License)
#
#  Copyright (c) 2008, Willow Garage, Inc.
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of the Willow Garage nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#******************************************************************* 

class PID(object):
    """ 
    A basic adapted PID class for a 7DoF robot.

    This class implements a generic structure that can be used to
    create a wide range of PID controllers. It can function
    independently or be subclassed to provide more specific controls
    based on a particular control loop.

    In particular, this class implements the standard pid equation:

    $command = p_{term} + i_{term} + d_{term} $

    where:

    $ p_{term} = p_{gain} * p_{error} $
    $ i_{term} = i_{gain} * i_{error} $
    $ d_{term} = d_{gain} * d_{error} $
    $ i_{error} = i_{error} + p_{error} * dt $
    $ d_{error} = (p_{error} - p_{error last}) / dt $

    given:

    $ p_{error} = p_{target} - p_{state} $.
    """

    def __init__(self, p_gain, i_gain, d_gain, i_min, i_max):
        """
        Constructor, zeros out Pid values when created and
        initialize Pid-gains and integral term limits. All gains are 
		7x7 matrices.

        Parameters:
          p_gain     The proportional gain.
          i_gain     The integral gain.
          d_gain     The derivative gain.
          i_min      The integral lower limit. 
          i_max      The integral upper limit.
        """
        # TODO Generalize this to num_dofs
        self.num_dofs = 6
        self.set_gains(p_gain, i_gain, d_gain, i_min, i_max)
        self.reset()
        self.i = 0

    def reset(self):
        """ Reset the state of this PID controller """
        self._p_error_last = np.zeros((self.num_dofs,1)) # Save position state for derivative
                                 # state calculation.
        self._p_error = np.zeros((self.num_dofs,1))  # Position error.
        self._d_error = np.zeros((self.num_dofs,1))  # Derivative error.
        self._i_error = np.zeros((self.num_dofs,1))  # Integator error.
        self._cmd = np.zeros((self.num_dofs,self.num_dofs))  # Command to send.
        self._last_time = None # Used for automatic calculation of dt.
        
    def set_gains(self, p_gain, i_gain, d_gain, i_min, i_max): 
        """ 
        Set PID gains for the controller. 

         Parameters:
          p_gain     The proportional gain.
          i_gain     The integral gain.
          d_gain     The derivative gain.
          i_min      The integral lower limit. 
          i_max      The integral upper limit.
        """ 
        self._p_gain = p_gain
        self._i_gain = i_gain
        self._d_gain = d_gain
        self._i_min = i_min
        self._i_max = i_max

    @property
    def p_gain(self):
        """ Read-only access to p_gain. """
        return self._p_gain

    @property
    def i_gain(self):
        """ Read-only access to i_gain. """
        return self._i_gain

    @property
    def d_gain(self):
        """ Read-only access to d_gain. """
        return self._d_gain

    @property
    def i_max(self):
        """ Read-only access to i_max. """
        return self._i_max

    @property
    def i_min(self):
        """ Read-only access to i_min. """
        return self._i_min

    @property
    def p_error(self):
        """ Read-only access to p_error. """
        return self._p_error

    @property
    def i_error(self):
        """ Read-only access to i_error. """
        return self._i_error

    @property
    def d_error(self):
        """ Read-only access to d_error. """
        return self._d_error

    @property
    def cmd(self):
        """ Read-only access to the latest command. """
        return self._cmd

    @property
    def last_time(self):
       """ Read-only access to the last time. """
       return self._last_time

    def __str__(self):
        """ String representation of the current state of the controller. """
        result = ""
        result += "p_gain:  " + str(self.p_gain) + "\n"
        result += "i_gain:  " + str(self.i_gain) + "\n"
        result += "d_gain:  " + str(self.d_gain) + "\n"
        result += "i_min:   " + str(self.i_min) + "\n"
        result += "i_max:   " + str(self.i_max) + "\n"
        result += "p_error: " + str(self.p_error) + "\n"
        result += "i_error: " + str(self.i_error) + "\n"
        result += "d_error: " + str(self.d_error) + "\n"
        result += "cmd:     " + str(self.cmd) + "\n"
        return result
        
    def update_PID(self, p_error, dt=None):

        """
        Update the PID loop with nonuniform time step size.

        Parameters:
          p_error  Error since last call (target - state)
          dt       Change in time since last call, in seconds, or None. 
                   If dt is None, then the system clock will be used to 
                   calculate the time since the last update. 
        """

        if dt == None:
            cur_time = time.time()
            if self._last_time is None:
                self._last_time = cur_time 
            dt = cur_time - self._last_time
            self._last_time = cur_time

        self._p_error = p_error
        if dt == 0 or math.isnan(dt) or math.isinf(dt):
            return np.zeros((self.num_dofs,self.num_dofs))

        # Calculate proportional contribution to command
        p_term = self._p_gain * self._p_error
        # p_str = np.array2string(p_term)
        # logger.info(f"p: {p_str}")
		
        # Calculate the integral error
        self._i_error += dt * self._p_error 
        
        # Calculate integral contribution to command
        i_term = self._i_gain * self._i_error
        # i_str = np.array2string(i_term)
        # logger.info(f"i: {i_str}")
        
        # Calculate the derivative error
        self._d_error = (self._p_error - self._p_error_last) / dt
        self._p_error_last = self._p_error

        # Calculate derivative contribution to command 
        d_term = self._d_gain * self._d_error
        # d_str = np.array2string(d_term)
        # logger.info(f"d: {d_str}\n")
        
        self._cmd = p_term + i_term + d_term
        # self._cmd = np.eye(self.num_dofs) * dt * p_error
        return self._cmd
