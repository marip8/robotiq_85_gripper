"""
*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2021, PickNik Inc
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of PickNik Inc nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************
 Author: Boston Cleek
 File:   single_robotiq_85_action_server
 Brief:  Action server for Robotiq 85 communication
 Platform: Linux/ROS Foxy
"""

import threading
import time
import numpy as np
from math import fabs

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType

from robotiq_85_driver.driver.robotiq_85_gripper import Robotiq85Gripper
from sensor_msgs.msg import JointState
from control_msgs.action import GripperCommand


class Robotiq85ActionServer(Node):
    def __init__(self):
        super().__init__('single_robotiq_85_action_server')

        # put a leep so it can connect to the force/torque sensor first
        time.sleep(5.0)

        self.declare_parameter('timeout', 2.0)            
        self.declare_parameter('position_tolerance', 0.005) 
        self.declare_parameter('gripper_speed', 0.0565)      
        self.declare_parameter('comport', '/dev/ttyUSB0')
        self.declare_parameter('baud', '115200')

        num_grippers = 1
        self._timeout = self.get_parameter('timeout').get_parameter_value().double_value
        self._position_tolerance = self.get_parameter('position_tolerance').get_parameter_value().double_value
        self._gripper_speed = self.get_parameter('gripper_speed').get_parameter_value().double_value
        self._comport = self.get_parameter('comport').get_parameter_value().string_value
        self._baud = self.get_parameter('baud').get_parameter_value().string_value
        
        self._max_pos = 0.085
        self._min_pos = 0.0
        self._max_vel = 0.013
        self._min_vel = 0.1
        self._min_force = 5.0
        self._max_force = 220.0 

        self._gripper_speed = self._clamp_cmd(self._gripper_speed, self._min_vel, self._max_vel)

        # self.get_logger().info("Timeout: %f, Comport: %s, Baud rate: %s"% (self._timeout, self._comport, self._baud))

        self._gripper = Robotiq85Gripper(num_grippers, self._comport, self._baud)

        if not self._gripper.init_success:
            self.get_logger().error("Unable to open commport to %s: " % self._comport)
            return

        self._gripper_joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self._action_server = ActionServer(
            self,
            GripperCommand,
            '/gripper/gripper_action',
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            execute_callback=self._execute_callback,
            callback_group=MutuallyExclusiveCallbackGroup()) # process one goal at a time

        self._prev_js_pos = 0.0
        self._prev_js_time = self.get_time()
        self._driver_state = 0
        self._driver_ready = False

        success = True
        success &= self._gripper.process_stat_cmd(0)
        if not success:
            self.get_logger().error("Failed to contact gripper")
            return

        # Publish joint states at 100Hz
        self.joints_timer = self.create_timer(0.01, self._joints_timer_callback)

        self._start_up_procedure()
        self.get_logger().info("Gripper ready for commands")


    def get_time(self):
        time_msg = self.get_clock().now().to_msg()
        return float(time_msg.sec) + (float(time_msg.nanosec) * 1e-9)


    def shutdown(self):
        self.get_logger().info("Shutdown gripper")
        self._gripper.shutdown()


    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()


    def _clamp_cmd(self, cmd, lower, upper):
        if (cmd < lower):
            return lower
        elif (cmd > upper):
            return upper
        else:
            return cmd


    def _start_up_procedure(self):
        self.get_logger().info('Gripper start up procedure')
        last_time = self.get_time()

        while rclpy.ok():
            # self.get_logger().info('driver state: %i'% self._driver_state)
            dt = self.get_time() - last_time
            if (self._driver_state == 0):
                if (dt < 0.5):
                    self._gripper.deactivate_gripper(0)
                    # self.get_logger().info('Deactivate_gripper')
                else:
                    self._driver_state = 1

            elif (self._driver_state == 1):
                grippers_activated = True
                self._gripper.activate_gripper(0)
                grippers_activated &= self._gripper.is_ready(0)
                # self.get_logger().info('Activate gripper and status: %r'% grippers_activated)
                
                if (grippers_activated):
                    self._driver_state = 2

            elif (self._driver_state == 2):
                self._driver_ready = True  
                # self.get_logger().info('Gripper driver is ready')
                break


    def _goal_callback(self, goal_request):
        self.get_logger().info('Gripper received goal request')
        return GoalResponse.ACCEPT


    def _cancel_callback(self, goal_handle):
        self.get_logger().info('Gripper received cancel request')
        return CancelResponse.ACCEPT


    def _execute_callback(self, goal_handle):
        self.get_logger().info('Gripper executing goal...')

        position_goal = self._clamp_cmd(goal_handle.request.command.position, self._min_pos, self._max_pos)
        force_goal = self._clamp_cmd(goal_handle.request.command.max_effort, self._min_force, self._max_force)

        self.get_logger().info('position goal: %f'% position_goal)
        self.get_logger().info('force goal: %f'% force_goal)

        # Send goal to gripper 
        self._gripper.goto(dev=0, pos=position_goal, vel=self._gripper_speed, force=force_goal)

        thread = threading.Thread(target=rclpy.spin, args=(self, ), daemon=True)
        thread.start()

        feedback_msg = GripperCommand.Feedback()
        result_msg = GripperCommand.Result()
        result_msg.reached_goal = False

        # update at 100Hz
        rate = self.create_rate(100)
        start_time = self.get_time()

        while rclpy.ok():
            dt = self.get_time() - start_time
            # print("cb:", dt)
            
            if not (dt < self._timeout):
                self.get_logger().warn('Gripper timeout reached')
                break

            if not self._driver_ready:
                self.get_logger().warn('Driver not ready')
                break

            success = True
            success &= self._gripper.process_act_cmd(0)
            success &= self._gripper.process_stat_cmd(0)
            if not success:
                self.get_logger().error("Failed to contact gripper")
            else:
                feedback_msg.position = self._gripper.get_pos(0)
                # feedback_msg.effort = self._gripper.get_current(0)
                feedback_msg.stalled = self._gripper.is_stopped(0)
                # feedback_msg.stalled = self._gripper.object_detected(0)

                # Position tolerance achieved or object grasped
                if (fabs(position_goal - feedback_msg.position) < self._position_tolerance or self._gripper.object_detected(0)):
                    feedback_msg.reached_goal = True
                    self.get_logger().info('Goal achieved: %r'% feedback_msg.reached_goal)

                goal_handle.publish_feedback(feedback_msg)

                if feedback_msg.reached_goal:
                    goal_handle.succeed()
                    break;
  
            rate.sleep()

        thread.join()
        result_msg = feedback_msg
        return result


    def _joints_timer_callback(self):
        # TODO: add effort to message
        # TODO: check self._driver_ready and self._gripper.process_stat_cmd(0)
        msg = JointState()
        msg.header.frame_id = ''
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['robotiq_85_left_knuckle_joint']
        pos = np.clip(0.8 - ((0.8/0.085) * self._gripper.get_pos(0)), 0., 0.8)
        msg.position = [pos]
        dt = self.get_time() - self._prev_js_time
        self._prev_js_time = self.get_time()
        msg.velocity = [(pos - self._prev_js_pos)/dt]
        self._prev_js_pos = pos
        self._gripper_joint_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    action_server = Robotiq85ActionServer()

    executor = MultiThreadedExecutor()
    rclpy.spin(action_server, executor=executor)
    # rclpy.spin(action_server)

    action_server.shutdown()
    action_server.destroy()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
