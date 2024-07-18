# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException

from hri_body_detect.multibody_detector import MultibodyDetector

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import ParameterDescriptor

from rclpy.lifecycle import Node, LifecycleState, TransitionCallbackReturn
from lifecycle_msgs.msg import State


# body detection processing time in ms triggering a diagnostic warning
BODY_DETECTION_PROC_TIME_WARN = 1000.

# diagnostic period
DIAGNOSTIC_PERIOD = 1.


class MultibodyNode(Node):
    """Node for detecting multiple bodies."""

    def __init__(self):
        super().__init__("hri_body_detect")
        self.declare_parameter(
            "use_depth",
            False,
            ParameterDescriptor(
                description="Whether or not using depth to process the bodies position.")
        )
        self.declare_parameter(
            "stickman_debug",
            False,
            ParameterDescriptor(
                description="Whether or not using the stickman debugging visualization."))

        self.declare_parameter(
            "detection_conf_thresh",
            0.5,
            ParameterDescriptor(
                description="Threshold to apply to the mediapipe pose detection."))

        self.declare_parameter(
            "use_cmc",
            False,
            ParameterDescriptor(
                description="Whether or not to enable camera motion compensation in the tracker."))

        self.get_logger().info('State: Unconfigured.')

    def __del__(self):
        state = self._state_machine.current_state
        self.on_shutdown(LifecycleState(state_id=state[0], label=state[1]))

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.internal_cleanup()
        self.get_logger().info('State: Unconfigured.')
        return super().on_cleanup(state)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.use_depth = self.get_parameter("use_depth").value
        self.stickman_debug = self.get_parameter("stickman_debug").value
        self.detection_conf_thresh = self.get_parameter("detection_conf_thresh").value
        self.use_cmc = self.get_parameter("use_cmc").value

        self.detector = MultibodyDetector(self,
                                          self.use_depth,
                                          self.stickman_debug,
                                          self.detection_conf_thresh,
                                          self.use_cmc)

        self.get_logger().info('State: Inactive.')
        return super().on_configure(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.internal_deactivate()
        self.get_logger().info('State: Inactive.')
        return super().on_deactivate(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.diag_timer = self.create_timer(DIAGNOSTIC_PERIOD, self.do_diagnostics)
        self.diag_pub = self.create_publisher(DiagnosticArray,
                                              "/diagnostics",
                                              1)

        self.get_logger().info('State: Active.')
        return super().on_activate(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        if state.state_id == State.PRIMARY_STATE_ACTIVE:
            self.internal_deactivate()
        self.get_logger().info('State: Finalized.')
        return super().on_shutdown(state)

    def internal_cleanup(self):
        del self.detector

    def internal_deactivate(self):
        self.destroy_timer(self.diag_timer)
        self.destroy_publisher(self.diag_pub)

    def do_diagnostics(self):
        """Perform diagnostic operations."""
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()

        proc_time = self.detector.get_proc_time().nanoseconds/1e9

        msg = DiagnosticStatus(name="Social perception: Body analysis: Skeleton extraction",
                               hardware_id="none")

        if self.detector.check_timeout():
            msg.level = DiagnosticStatus.ERROR
            msg.message = "Body detection process not responding"
        elif proc_time > BODY_DETECTION_PROC_TIME_WARN:
            msg.level = DiagnosticStatus.WARN
            msg.message = "Body detection processing is slow"
        else:
            msg.level = DiagnosticStatus.OK

        msg.values = [
            KeyValue(key="Package name", value='hri_body_detect'),
            KeyValue(key="Currently detected bodies",
                     value=str(len(self.detector.bodies))),
            KeyValue(key="Detection processing time",
                     value="{:.2f}".format(proc_time * 1000) + "ms"),
        ]

        arr.status = [msg]
        self.diag_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)

    node = MultibodyNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()


if __name__ == "__main__":
    main()
