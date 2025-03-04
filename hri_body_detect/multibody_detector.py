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


import mediapipe as mp
import io
import psutil
from ikpy import chain
from hri_body_detect.jointstate import compute_jointstate, \
    HUMAN_JOINT_NAMES
from hri_body_detect.rs_to_depth import rgb_to_xyz, DepthComputationError
from hri_body_detect.urdf_generator import make_urdf_human
from hri_body_detect.one_euro_filter import OneEuroFilter
from hri_body_detect.face_pose_estimation import face_pose_estimation
from hri_body_detect.BoTSORT.mc_bot_sort import BoTSORT
import numpy as np
import copy
import subprocess
import threading
import random
import ament_index_python
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import rclpy
from rclpy.node import Node
import tf2_ros
import tf_transformations
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import qos_profile_sensor_data

from builtin_interfaces.msg import Time as TimeInterface
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from hri_msgs.msg import Skeleton2D, NormalizedPointOfInterest2D, \
    NormalizedRegionOfInterest2D, IdsList, Gesture
from geometry_msgs.msg import TwistStamped, PointStamped, TransformStamped

from google.protobuf.pyext._message import RepeatedCompositeContainer

from cv_bridge import CvBridge
import cv2

directory = ament_index_python.get_package_share_directory('hri_body_detect')

# One Euro Filter parameters
BETA_POSITION = 0.05
D_CUTOFF_POSITION = 0.5
MIN_CUTOFF_POSITION = 0.3
BETA_VELOCITY = 0.2
D_CUTOFF_VELOCITY = 0.2
MIN_CUTOFF_VELOCITY = 0.5

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Mediapipe parameters
MP_MODEL_PATH = directory + '/weights/pose_landmarker_full.task'
MP_GESTURES_MODEL_PATH = directory + '/weights/gesture_recognizer.task'
MP_MAX_NUM_PEOPLE = 5
MP_HOL_TRACKING_CONFIDENCE = 0.5

# Mediapipe 2D keypoint order:
# ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
#   'right_eye_inner', 'right_eye', 'right_eye_outer',
#   'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
#   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
#   'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
#   'left_index', 'right_index', 'left_thumb', 'right_thumb',
#   'left_hip', 'right_hip', 'left_knee', 'right_knee',
#   'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
#   'left_foot_index', 'right_foot_index']

# Mediapipe 2D skeleton indexing

MP_NOSE = 0
MP_LEFT_EYE = 2
MP_LEFT_EAR = 7
MP_LEFT_MOUTH = 9
MP_LEFT_SHOULDER = 11
MP_LEFT_ELBOW = 13
MP_LEFT_WRIST = 15
MP_LEFT_HIP = 23
MP_LEFT_KNEE = 25
MP_LEFT_ANKLE = 27
MP_LEFT_FOOT = 31
MP_RIGHT_EYE = 5
MP_RIGHT_EAR = 8
MP_RIGHT_MOUTH = 9
MP_RIGHT_SHOULDER = 12
MP_RIGHT_ELBOW = 14
MP_RIGHT_WRIST = 16
MP_RIGHT_HIP = 24
MP_RIGHT_KNEE = 26
MP_RIGHT_ANKLE = 28
MP_RIGHT_FOOT = 32

# ROS4HRI to Mediapipe 2D skeleton indexing conversion table

ros4hri_to_mediapipe = [None] * 18

ros4hri_to_mediapipe[Skeleton2D.NOSE] = MP_NOSE
ros4hri_to_mediapipe[Skeleton2D.LEFT_EYE] = MP_LEFT_EYE
ros4hri_to_mediapipe[Skeleton2D.LEFT_EAR] = MP_LEFT_EAR
ros4hri_to_mediapipe[Skeleton2D.LEFT_SHOULDER] = MP_LEFT_SHOULDER
ros4hri_to_mediapipe[Skeleton2D.LEFT_ELBOW] = MP_LEFT_ELBOW
ros4hri_to_mediapipe[Skeleton2D.LEFT_WRIST] = MP_LEFT_WRIST
ros4hri_to_mediapipe[Skeleton2D.LEFT_HIP] = MP_LEFT_HIP
ros4hri_to_mediapipe[Skeleton2D.LEFT_KNEE] = MP_LEFT_KNEE
ros4hri_to_mediapipe[Skeleton2D.LEFT_ANKLE] = MP_LEFT_ANKLE
ros4hri_to_mediapipe[Skeleton2D.RIGHT_EYE] = MP_RIGHT_EYE
ros4hri_to_mediapipe[Skeleton2D.RIGHT_EAR] = MP_RIGHT_EAR
ros4hri_to_mediapipe[Skeleton2D.RIGHT_SHOULDER] = MP_RIGHT_SHOULDER
ros4hri_to_mediapipe[Skeleton2D.RIGHT_ELBOW] = MP_RIGHT_ELBOW
ros4hri_to_mediapipe[Skeleton2D.RIGHT_WRIST] = MP_RIGHT_WRIST
ros4hri_to_mediapipe[Skeleton2D.RIGHT_HIP] = MP_RIGHT_HIP
ros4hri_to_mediapipe[Skeleton2D.RIGHT_KNEE] = MP_RIGHT_KNEE
ros4hri_to_mediapipe[Skeleton2D.RIGHT_ANKLE] = MP_RIGHT_ANKLE

# Mediapipe to hri_msgs Gesture constants
mediapipe_gesture_to_hri = {}
mediapipe_gesture_to_hri["Hands_On_Face"] = Gesture.HANDS_ON_FACE
mediapipe_gesture_to_hri["Arms_Crossed"] = Gesture.ARMS_CROSSED
mediapipe_gesture_to_hri["Left_Hand_Raised"] = Gesture.LEFT_HAND_RAISED
mediapipe_gesture_to_hri["Right_Hand_Raised"] = Gesture.RIGHT_HAND_RAISED
mediapipe_gesture_to_hri["Both_Hands_Raised"] = Gesture.BOTH_HANDS_RAISED
mediapipe_gesture_to_hri["Waving"] = Gesture.WAVING
mediapipe_gesture_to_hri["Closed_Fist"] = Gesture.CLOSED_FIST
mediapipe_gesture_to_hri["Open_Palm"] = Gesture.OPEN_PALM
mediapipe_gesture_to_hri["Pointing_Up"] = Gesture.POINTING_UP
mediapipe_gesture_to_hri["Thumb_Down"] = Gesture.THUMB_DOWN
mediapipe_gesture_to_hri["Thumb_Up"] = Gesture.THUMB_UP
mediapipe_gesture_to_hri["Victory"] = Gesture.VICTORY
mediapipe_gesture_to_hri["ILoveYou"] = Gesture.LOVE
mediapipe_gesture_to_hri["Other"] = Gesture.OTHER
mediapipe_gesture_to_hri["None"] = Gesture.OTHER
mediapipe_gesture_to_hri[None] = Gesture.OTHER

mediapipe_handedness_to_hri = {}
mediapipe_handedness_to_hri["Right"] = Gesture.RIGHT
mediapipe_handedness_to_hri["Left"] = Gesture.LEFT
mediapipe_handedness_to_hri["Independent"] = Gesture.INDEPENDENT
mediapipe_handedness_to_hri["None"] = Gesture.INDEPENDENT
mediapipe_handedness_to_hri[None] = Gesture.INDEPENDENT

# Mediapipe face mesh indexing (partial)
FM_NOSE = 1
FM_MOUTH_CENTER = 13
FM_RIGHT_EYE = 159
FM_RIGHT_EAR_TRAGION = 234
FM_LEFT_EYE = 386
FM_LEFT_EAR_TRAGION = 454

# body detection processing time in ms signalling a timeout error
BODY_DETECTION_PROC_TIMEOUT_ERROR = 5000.


class BoTTrackerArgs:
    def __init__(self, **kwargs):
        # Not relevant, we do not have a confidence score for the whole skeleton
        self.track_high_thresh = kwargs.pop("track_high_thresh", 0.3)
        # Not relevant, we do not have a confidence score for the whole skeleton
        self.track_low_thresh = kwargs.pop("track_low_thresh", 0.05)
        # Not relevant, we do not have a confidence score for the whole skeleton
        self.new_track_thresh = kwargs.pop("new_track_thresh", 0.4)
        self.track_buffer = kwargs.pop("track_buffer", 100)
        self.match_thresh = kwargs.pop("match_thresh", 0.7)
        self.aspect_ratio_thresh = kwargs.pop("aspect_ratio_thresh", 1.6)
        self.min_box_area = kwargs.pop("min_box_area", 10)
        # one of "none", "orb", "sparseOptFlow"
        self.cmc_method = kwargs.pop("cmc-method", "none")
        self.frame_rate = kwargs.pop("frame_rate", 20)


def _normalized_to_pixel_coordinates(
        normalized_x: float,
        normalized_y: float,
        image_width: int,
        image_height: int) -> (float, float):
    """Transform normalized image coordinates into pixel coordinates."""
    x_px = min(np.floor(normalized_x * image_width), image_width - 1)
    y_px = min(np.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def _make_2d_skeleton_msg(header: Header,
                          pose_2d: list[dict[float, float, float, float]]) -> Skeleton2D:
    """
    Take the mediapipe output and transform it into a Skeleton2D message.

    Transform skeleton 2d coordinates coming from
    mediapipe pose detection into ROS4HRI 2d skeleton
    format (Skeleton2D).
    """
    skel = Skeleton2D()
    skel.header = header

    for idx, human_joint in enumerate(ros4hri_to_mediapipe):
        if human_joint is not None:
            skel.skeleton[idx].x = pose_2d[human_joint].get('x')
            skel.skeleton[idx].y = pose_2d[human_joint].get('y')
            skel.skeleton[idx].c = pose_2d[human_joint].get('visibility')

    # There is no Neck landmark in Mediapipe pose estimation
    # However, we can think of the neck point as the average
    # point between left and right shoulder.
    skel.skeleton[Skeleton2D.NECK] = NormalizedPointOfInterest2D()
    skel.skeleton[Skeleton2D.NECK].x = (
        skel.skeleton[Skeleton2D.LEFT_SHOULDER].x
        + skel.skeleton[Skeleton2D.RIGHT_SHOULDER].x
    )/2
    skel.skeleton[Skeleton2D.NECK].y = (
        skel.skeleton[Skeleton2D.LEFT_SHOULDER].y
        + skel.skeleton[Skeleton2D.RIGHT_SHOULDER].y
    )/2
    skel.skeleton[Skeleton2D.NECK].c = min(skel.skeleton[Skeleton2D.LEFT_SHOULDER].c,
                                           skel.skeleton[Skeleton2D.RIGHT_SHOULDER].c)

    return skel


def _get_bounding_box_limits(landmarks: RepeatedCompositeContainer) -> (float, float,
                                                                        float, float):
    """Return limit bounding box coordinates given a list of landmarks."""
    x_max = 0.0
    y_max = 0.0
    x_min = 1.0
    y_min = 1.0
    # for result in results:
    for data_point in landmarks:
        if x_max < data_point.x:
            x_max = data_point.x
        if y_max < data_point.y:
            y_max = data_point.y
        if x_min > data_point.x:
            x_min = data_point.x
        if y_min > data_point.y:
            y_min = data_point.y

    return x_min, y_min, x_max, y_max


def _builtin_time_to_secs(time: TimeInterface) -> float:
    """Transform builtin_interface.Time to float."""
    return time.sec+(time.nanosec/1e9)


def _find_optimal_coupling(A, B):
    """Hungarian algorithm."""
    distance_matrix = cdist(A, B, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    return row_ind, col_ind


@dataclass
class SingleHand:
    """Class representing a single hand."""

    landmarks: list
    handedness: str
    gesture: str


class SingleBody:
    """Class managing the single body processing."""

    def __init__(self,
                 node: Node,
                 use_depth: bool,
                 stickman_debug: bool,
                 body_id: str,
                 camera_info: CameraInfo,
                 img_width: int,
                 img_height: int,
                 depth_encoding: str):

        self.use_depth = use_depth
        self.stickman_debug = stickman_debug
        self.node = node
        self.depth_encoding = depth_encoding

        self.last_stamp = camera_info.header.stamp

        self.img_width = img_width
        self.img_height = img_height

        self.K = np.zeros((3, 3), np.float32)
        self.K[0][0:3] = camera_info.k[0:3]
        self.K[1][0:3] = camera_info.k[3:6]
        self.K[2][0:3] = camera_info.k[6:9]

        self.f_x = self.K[0][0]
        self.f_y = self.K[1][1]
        self.c_x = self.K[0][2]
        self.c_y = self.K[1][2]

        self.calibrated_camera = np.any(self.K)

        self.valid_trans_vec = False

        # self.skeleton_to_set = True

        self.body_id = body_id

        self.js_topic = "/humans/bodies/" + body_id + "/joint_states"
        skel_topic = "/humans/bodies/" + body_id + "/skeleton2d"
        self.urdf_topic = "/humans/bodies/" + body_id + "/urdf"

        self.skel_pub = self.node.create_publisher(Skeleton2D,
                                                   skel_topic,
                                                   1)

        self.js_pub = self.node.create_publisher(JointState,
                                                 self.js_topic,
                                                 1)

        self.roi_pub = self.node.create_publisher(
            NormalizedRegionOfInterest2D,
            "/humans/bodies/"+body_id+"/roi",
            1)

        filtered_position_topic = "/humans/bodies/"+body_id+"/position"
        self.body_filtered_position_pub = self.node.create_publisher(
            PointStamped,
            filtered_position_topic,
            1)

        twist_topic = "/humans/bodies/"+body_id+"/velocity"
        self.velocity_pub = self.node.create_publisher(
            TwistStamped,
            twist_topic,
            1)

        gesture_topic = "/humans/bodies/"+body_id+"/gesture"
        self.gesture_pub = self.node.create_publisher(
            Gesture,
            gesture_topic,
            1)

        self.body_position_estimation = [None] * 3
        # trans_vec ==> vector representing the translational component
        # of the homoegenous transform obtained solving the PnP problem
        # between the camera optical frame and the face frame.
        # self.trans_vec = [None] * 3
        # self.valid_trans_vec = False

        self.body_filtered_position = [None] * 3
        self.body_filtered_position_prev = [None] * 3
        self.body_vel_estimation = [None] * 3
        self.body_vel_estimation_filtered = [None] * 3

        self.position_msg = PointStamped()
        self.velocity_msg = TwistStamped()
        self.velocity_msg.header.frame_id = "body_"+body_id

        self.tb = tf2_ros.TransformBroadcaster(self.node)

        self.one_euro_filter = [None] * 3
        self.one_euro_filter_dot = [None] * 3

        self.left_hand = None
        self.right_hand = None

        self.skeleton_generation()

    def skeleton_generation(self):
        """
        Generate a URDF model for this body.

        This function generates the URDF model associated to
        the body handled by the FullbodyDetector object.
        Additionally, it spawns a new robot_state_publisher
        node handling the transformations associated with
        the newly generated URDF model.
        """
        self.urdf = make_urdf_human(self.body_id)
        self.node.get_logger().info("Setting URDF description for body"
                                    "<%s> (param name: human_description_%s)" % (
                                        self.body_id, self.body_id))
        self.human_description = "human_description_%s" % self.body_id

        self.urdf_file = io.StringIO(self.urdf)
        self.r_arm_chain = chain.Chain.from_urdf_file(
            self.urdf_file,
            base_elements=[
                "r_y_shoulder_%s" % self.body_id],
            base_element_type="joint",
            active_links_mask=[False, True, True, True, True, False])
        self.urdf_file.seek(0)
        self.l_arm_chain = chain.Chain.from_urdf_file(
            self.urdf_file,
            base_elements=[
                "l_y_shoulder_%s" % self.body_id],
            base_element_type="joint",
            active_links_mask=[False, True, True, True, True, False])
        self.urdf_file.seek(0)
        self.r_leg_chain = chain.Chain.from_urdf_file(
            self.urdf_file,
            base_elements=[
                "r_y_hip_%s" % self.body_id],
            base_element_type="joint",
            active_links_mask=[False, True, True, True, True, False])
        self.urdf_file.seek(0)
        self.l_leg_chain = chain.Chain.from_urdf_file(
            self.urdf_file,
            base_elements=[
                "l_y_hip_%s" % self.body_id],
            base_element_type="joint",
            active_links_mask=[False, True, True, True, True, False])

        self.ik_chains = {}  # maps a body id to the IKpy chains
        self.ik_chains[self.body_id] = [
            self.r_arm_chain,
            self.l_arm_chain,
            self.r_leg_chain,
            self.l_leg_chain
        ]

        self.node.get_logger().info(
            "Spawning a instance of robot_state_publisher for this body...")
        cmd = ["ros2", "run", "robot_state_publisher", "robot_state_publisher",
               "--ros-args", "-r", "__node:=robot_state_publisher_body_%s" % self.body_id,
               "-p",
               "robot_description:=%s" % self.urdf,
               "-r",
               "joint_states:="+self.js_topic,
               "-r",
               "robot_description:="+self.urdf_topic
               ]
        self.proc = subprocess.Popen(cmd,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.STDOUT)

    def unregister(self):
        """Kill the robot state publisher."""
        process = psutil.Process(self.proc.pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()
        self.proc.wait()
        self.node.get_logger().info('unregistered '+self.body_id)

    def face_to_body_position_estimation(self,
                                         skel_msg: Skeleton2D) -> (float, float, float):
        """Estimates body pose from face pose estimation."""
        body_px = [(skel_msg.skeleton[Skeleton2D.LEFT_HIP].x
                    + skel_msg.skeleton[Skeleton2D.RIGHT_HIP].x)
                   / 2,
                   (skel_msg.skeleton[Skeleton2D.LEFT_HIP].y
                    + skel_msg.skeleton[Skeleton2D.RIGHT_HIP].y)
                   / 2]
        body_px = _normalized_to_pixel_coordinates(body_px[0],
                                                   body_px[1],
                                                   self.img_width,
                                                   self.img_height)
        if body_px == [0, 0]:
            return [0, 0, 0]
        else:
            d_x = np.sqrt((self.trans_vec[0]/1000)**2
                          + (self.trans_vec[1]/1000)**2
                          + (self.trans_vec[2]/1000)**2)

            x = body_px[0]-self.c_x
            y = body_px[1]-self.c_y

            Z = self.f_x*d_x/(np.sqrt(x**2 + self.f_x**2))
            X = x*Z/self.f_x
            Y = y*Z/self.f_y
            return [X, Y, Z]

    def create_transform(self,
                         trasl: list[float, float, float],
                         rot: list[float, float, float, float],
                         stamp: TimeInterface,
                         child_frame_id: str,
                         frame_id: str) -> TransformStamped:
        """Take trans/rot (as lists) and return a TransformStamped object."""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = frame_id
        t.child_frame_id = child_frame_id
        t.transform.translation.x = trasl[0]
        t.transform.translation.y = trasl[1]
        t.transform.translation.z = trasl[2]
        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = rot[3]

        return t

    def stickman_debugging(self,
                           theta: float,
                           torso: list[float, float, float],
                           torso_res: list[float, float, float],
                           l_shoulder: list[float, float, float],
                           r_shoulder: list[float, float, float],
                           l_elbow: list[float, float, float],
                           r_elbow: list[float, float, float],
                           l_wrist: list[float, float, float],
                           r_wrist: list[float, float, float],
                           l_ankle: list[float, float, float],
                           r_ankle: list[float, float, float],
                           header: Header):
        """
        Publish stickman debugging transforms.

        Stickman debugging: publishing body parts tf frames directly
        using the estimation obtained from Mediapipe, that is,
        the ones used as an input for the IK/FK process
        """
        self.tb.sendTransform(
            self.create_transform(
                (-torso[1]+torso_res[0], torso[2], torso[0]+torso_res[2]),
                tf_transformations.quaternion_from_euler(
                    np.pi/2,
                    -theta,
                    0
                ),
                header.stamp,
                "mediapipe_torso_"+self.body_id,
                header.frame_id
            )
        )
        self.tb.sendTransform(
            self.create_transform(
                (0.0, 0.0, 0.605),
                tf_transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "our_torso_"+self.body_id,
                "mediapipe_torso_"+self.body_id
            )
        )
        self.tb.sendTransform(
            self.create_transform(
                (l_shoulder[0], l_shoulder[1], l_shoulder[2]),
                tf_transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "left_shoulder_"+self.body_id,
                "our_torso_"+self.body_id
            )
        )
        self.tb.sendTransform(
            self.create_transform(
                (r_shoulder[0], r_shoulder[1], r_shoulder[2]),
                tf_transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "right_shoulder_"+self.body_id,
                "our_torso_"+self.body_id
            )
        )
        self.tb.sendTransform(
            self.create_transform(
                (l_elbow[0]-l_shoulder[0],
                 l_elbow[1]-l_shoulder[1],
                 l_elbow[2]-l_shoulder[2]
                 ),
                tf_transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "left_elbow_"+self.body_id,
                "left_shoulder_"+self.body_id
            )
        )
        self.tb.sendTransform(
            self.create_transform(
                (r_elbow[0]-r_shoulder[0],
                 r_elbow[1]-r_shoulder[1],
                 r_elbow[2]-r_shoulder[2]
                 ),
                tf_transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "right_elbow_"+self.body_id,
                "right_shoulder_"+self.body_id
            )
        )
        self.tb.sendTransform(
            self.create_transform(
                (l_wrist[0]-l_elbow[0],
                 l_wrist[1]-l_elbow[1],
                 l_wrist[2]-l_elbow[2]),
                tf_transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "left_wrist_"+self.body_id,
                "left_elbow_"+self.body_id
            )
        )
        self.tb.sendTransform(
            self.create_transform(
                (r_wrist[0]-r_elbow[0],
                 r_wrist[1]-r_elbow[1],
                 r_wrist[2]-r_elbow[2]
                 ),
                tf_transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "right_wrist_"+self.body_id,
                "right_elbow_"+self.body_id
            )
        )
        self.tb.sendTransform(
            self.create_transform(
                (l_ankle[0],
                 l_ankle[1],
                 l_ankle[2]
                 ),
                tf_transformations.quaternion_from_euler(
                    0,
                    0,
                    0),
                header.stamp,
                "left_ankle_"+self.body_id,
                "mediapipe_torso_"+self.body_id
            )
        )
        self.tb.sendTransform(
            self.create_transform(
                (r_ankle[0],
                 r_ankle[1],
                 r_ankle[2]
                 ),
                tf_transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "right_ankle_"+self.body_id,
                "mediapipe_torso_"+self.body_id
            )
        )

    def make_jointstate(
            self,
            body_id: str,
            pose_3d: list[dict[float, float, float, float]],
            pose_2d: list[dict[float, float, float, float]],
            header: Header) -> JointState:
        """
        Handle the inverse kinematic process.

        Given the current human pose, as detected per the holistic
        body detection, this function computes the joint values
        for the kinematic human model.
        """
        js = JointState()
        js.header = copy.copy(header)
        js.name = [jn + "_%s" % body_id for jn in HUMAN_JOINT_NAMES]

        torso = np.array([
            -(
                pose_3d[MP_LEFT_HIP].get('z')
                + pose_3d[MP_RIGHT_HIP].get('z')
            )
            / 2,
            (
                pose_3d[MP_LEFT_HIP].get('x')
                + pose_3d[MP_RIGHT_HIP].get('x')
            )
            / 2,
            -(
                pose_3d[MP_LEFT_HIP].get('y')
                + pose_3d[MP_RIGHT_HIP].get('y')
            )
            / 2
        ])
        l_shoulder = np.array([
            -pose_3d[MP_LEFT_SHOULDER].get('z'),
            pose_3d[MP_LEFT_SHOULDER].get('x'),
            -pose_3d[MP_LEFT_SHOULDER].get('y')-0.605
        ])
        l_elbow = np.array([
            -pose_3d[MP_LEFT_ELBOW].get('z'),
            pose_3d[MP_LEFT_ELBOW].get('x'),
            -pose_3d[MP_LEFT_ELBOW].get('y')-0.605
        ])
        l_wrist = np.array([
            -pose_3d[MP_LEFT_WRIST].get('z'),
            pose_3d[MP_LEFT_WRIST].get('x'),
            -pose_3d[MP_LEFT_WRIST].get('y')-0.605
        ])
        l_ankle = np.array([
            -pose_3d[MP_LEFT_ANKLE].get('z'),
            pose_3d[MP_LEFT_ANKLE].get('x'),
            -pose_3d[MP_LEFT_ANKLE].get('y')
        ])
        r_shoulder = np.array([
            -pose_3d[MP_RIGHT_SHOULDER].get('z'),
            pose_3d[MP_RIGHT_SHOULDER].get('x'),
            -pose_3d[MP_RIGHT_SHOULDER].get('y')-0.605
        ])
        r_elbow = np.array([
            -pose_3d[MP_RIGHT_ELBOW].get('z'),
            pose_3d[MP_RIGHT_ELBOW].get('x'),
            -pose_3d[MP_RIGHT_ELBOW].get('y')-0.605
        ])
        r_wrist = np.array([
            -pose_3d[MP_RIGHT_WRIST].get('z'),
            pose_3d[MP_RIGHT_WRIST].get('x'),
            -pose_3d[MP_RIGHT_WRIST].get('y')-0.605
        ])
        r_ankle = np.array([
            -pose_3d[MP_RIGHT_ANKLE].get('z'),
            pose_3d[MP_RIGHT_ANKLE].get('x'),
            -pose_3d[MP_RIGHT_ANKLE].get('y')
        ])

        # depht and rotation #

        theta = np.arctan2(pose_3d[MP_RIGHT_HIP].get(
            'x'), -pose_3d[MP_RIGHT_HIP].get('z'))
        if self.use_depth:
            torso_px = _normalized_to_pixel_coordinates(
                (pose_2d[MP_LEFT_HIP].get('x') +
                 pose_2d[MP_RIGHT_HIP].get('x'))/2,
                (pose_2d[MP_LEFT_HIP].get('y') +
                 pose_2d[MP_RIGHT_HIP].get('y'))/2,
                self.img_width,
                self.img_height)
            torso_res = rgb_to_xyz(
                torso_px[0],
                torso_px[1],
                self.rgb_info,
                self.depth_info,
                self.depth_encoding,
                self.image_depth
            )
        elif self.body_position_estimation[0]:
            torso_res = self.body_position_estimation
        else:
            torso_res = np.array([0., 0., 0.])

        # Publishing tf transformations #

        t = float(header.stamp.sec)+(float(header.stamp.nanosec)/1e9)

        if not self.one_euro_filter[0] and self.use_depth:
            self.one_euro_filter[0] = OneEuroFilter(
                t,
                torso_res[2],
                beta=BETA_POSITION,
                d_cutoff=D_CUTOFF_POSITION,
                min_cutoff=MIN_CUTOFF_POSITION)
            self.one_euro_filter[1] = OneEuroFilter(
                t,
                torso_res[0],
                beta=BETA_POSITION,
                d_cutoff=D_CUTOFF_POSITION,
                min_cutoff=MIN_CUTOFF_POSITION)
            self.body_filtered_position[0] = torso_res[2]
            self.body_filtered_position[1] = torso_res[0]
        elif self.use_depth:
            self.body_filtered_position_prev[0] = \
                self.body_filtered_position[0]
            self.body_filtered_position_prev[1] = \
                self.body_filtered_position[1]
            self.body_filtered_position[0], t_e = \
                self.one_euro_filter[0](t, torso_res[2])
            self.body_filtered_position[1], _ = \
                self.one_euro_filter[1](t, torso_res[0])

            self.position_msg.point.z = self.body_filtered_position[0]
            self.position_msg.point.y = 0.0
            self.position_msg.point.x = self.body_filtered_position[1]
            self.position_msg.header.stamp = self.node.get_clock().now().to_msg()  # TODO
            self.position_msg.header.frame_id = header.frame_id
            self.body_filtered_position_pub.publish(self.position_msg)

            self.body_vel_estimation[0] = \
                (self.body_filtered_position[0]
                 - self.body_filtered_position_prev[0])/t_e
            self.body_vel_estimation[1] = \
                (self.body_filtered_position[1]
                 - self.body_filtered_position_prev[1])/t_e

            if not self.one_euro_filter_dot[0]:
                self.one_euro_filter_dot[0] = OneEuroFilter(
                    t,
                    self.body_vel_estimation[0],
                    beta=BETA_VELOCITY,
                    d_cutoff=D_CUTOFF_VELOCITY,
                    min_cutoff=MIN_CUTOFF_VELOCITY)
                self.one_euro_filter_dot[1] = OneEuroFilter(
                    t,
                    self.body_vel_estimation[1],
                    beta=BETA_VELOCITY,
                    d_cutoff=D_CUTOFF_VELOCITY,
                    min_cutoff=MIN_CUTOFF_VELOCITY)
            else:
                self.body_vel_estimation_filtered[0], _ = \
                    self.one_euro_filter_dot[0](t, self.body_vel_estimation[0])
                self.body_vel_estimation_filtered[1], _ = \
                    self.one_euro_filter_dot[1](t, self.body_vel_estimation[1])
                self.velocity_msg.twist.linear.x = \
                    -self.body_vel_estimation_filtered[0]
                self.velocity_msg.twist.linear.y = \
                    self.body_vel_estimation_filtered[1]
                self.velocity_pub.publish(self.velocity_msg)

        if not self.use_depth:
            translation = (torso_res[0], 0.0, torso_res[2])
        else:
            translation = (self.body_filtered_position[1],
                           0.0,
                           self.body_filtered_position[0])
        self.tb.sendTransform(
            self.create_transform(
                translation,
                tf_transformations.quaternion_from_euler(
                    np.pi/2,
                    -theta,
                    0
                ),
                header.stamp,
                "body_%s" % body_id,
                header.frame_id
            )
        )

        if self.stickman_debug:
            self.stickman_debugging(theta,
                                    torso,
                                    torso_res,
                                    l_shoulder,
                                    r_shoulder,
                                    l_elbow,
                                    r_elbow,
                                    l_wrist,
                                    r_wrist,
                                    l_ankle,
                                    r_ankle,
                                    header)

        js.position = compute_jointstate(
            self.ik_chains[body_id],
            torso,
            l_wrist,
            l_ankle,
            r_wrist,
            r_ankle
        )

        return js

    def process(self, pose_kpt, pose_world_kpt, bbox, header, rgb_info, depth_info, image_depth):

        self.last_stamp = header.stamp
        self.rgb_info = rgb_info
        self.depth_info = depth_info
        self.image_depth = image_depth

        if not self.use_depth and hasattr(self, "K"):
            # K = camera intrisic matrix. See method camera_info_callback
            #     to understand more about it
            for idx, landmark in enumerate(pose_kpt):

                if idx == MP_NOSE:
                    nose_tip = [landmark["x"], landmark["y"]]
                if idx == MP_LEFT_MOUTH:
                    mouth_left = [landmark["x"], landmark["y"]]
                if idx == MP_RIGHT_MOUTH:
                    mouth_right = [landmark["x"], landmark["y"]]
                if idx == MP_RIGHT_EYE:
                    right_eye = [landmark["x"], landmark["y"]]
                if idx == MP_RIGHT_EAR:
                    right_ear_tragion = [landmark["x"], landmark["y"]]
                if idx == MP_LEFT_EYE:
                    left_eye = [landmark["x"], landmark["y"]]
                if idx == MP_LEFT_EAR:
                    left_ear_tragion = [landmark["x"], landmark["y"]]

            mouth_center = [(mouth_left[0] + mouth_right[0])/2,
                            (mouth_left[1] + mouth_right[1])/2]

            points_2D = np.array([
                _normalized_to_pixel_coordinates(
                    nose_tip[0],
                    nose_tip[1],
                    self.img_width,
                    self.img_height),
                _normalized_to_pixel_coordinates(
                    right_eye[0],
                    right_eye[1],
                    self.img_width,
                    self.img_height),
                _normalized_to_pixel_coordinates(
                    left_eye[0],
                    left_eye[1],
                    self.img_width,
                    self.img_height),
                _normalized_to_pixel_coordinates(
                    mouth_center[0],
                    mouth_center[1],
                    self.img_width,
                    self.img_height),
                _normalized_to_pixel_coordinates(
                    right_ear_tragion[0],
                    right_ear_tragion[1],
                    self.img_width,
                    self.img_height),
                _normalized_to_pixel_coordinates(
                    left_ear_tragion[0],
                    left_ear_tragion[1],
                    self.img_width,
                    self.img_height)],
                dtype="double")

            self.trans_vec, self.angles = \
                face_pose_estimation(points_2D, self.K)

            if not self.trans_vec[0] \
                    or not self.trans_vec[1] \
                    or not self.trans_vec[2]:
                self.valid_trans_vec = False
            elif np.isnan(self.trans_vec[0]) \
                    or np.isnan(self.trans_vec[1]) \
                    or np.isnan(self.trans_vec[2]):
                if not self.calibrated_camera:
                    self.trans_vec = np.zeros(3)
                    self.valid_trans_vec = True
                else:
                    self.valid_trans_vec = False
            else:
                self.valid_trans_vec = True

        skel_msg = _make_2d_skeleton_msg(header, pose_kpt)
        if self.valid_trans_vec and not self.use_depth\
                and self.calibrated_camera:
            self.body_position_estimation = \
                self.face_to_body_position_estimation(skel_msg)
        elif self.valid_trans_vec and not self.use_depth\
                and not self.calibrated_camera:
            self.body_position_estimation = [0, 0, 0]
        elif not self.valid_trans_vec and not self.use_depth \
                and self.calibrated_camera:
            self.node.get_logger().error(
                "It was not possible to estimate body position.")
        if self.use_depth or self.valid_trans_vec:
            try:
                js = self.make_jointstate(
                    self.body_id,
                    pose_world_kpt,
                    pose_kpt,
                    header
                )
                self.js_pub.publish(js)
            except DepthComputationError as dce:
                self.node.get_logger().error(str(dce))

        self.skel_pub.publish(skel_msg)
        roi = NormalizedRegionOfInterest2D()
        roi.xmin = bbox[0]/self.img_width
        roi.ymin = bbox[1]/self.img_height
        roi.xmax = bbox[2]/self.img_width
        roi.ymax = bbox[3]/self.img_height
        roi.c = 0.5  # TODO: compute confidence from mediapipe landmarks confidence

        self.roi_pub.publish(roi)

        gesture = self.left_hand \
            if self.left_hand is not None and self.left_hand.gesture != "None" \
            else self.right_hand
        gesture_value = mediapipe_gesture_to_hri[gesture.gesture] \
            if gesture is not None \
            else mediapipe_gesture_to_hri[gesture]
        handedness_value = mediapipe_handedness_to_hri[gesture.handedness] \
            if gesture is not None and gesture.gesture != "None" \
            else mediapipe_handedness_to_hri["None"]

        gesture_msg = Gesture(
            header=header,
            gesture=gesture_value,
            handedness=handedness_value)

        self.gesture_pub.publish(gesture_msg)

    def __del__(self):
        self.node.destroy_publisher(self.roi_pub)
        self.node.destroy_publisher(self.skel_pub)
        self.node.destroy_publisher(self.js_pub)
        self.node.destroy_publisher(self.body_filtered_position_pub)
        self.node.destroy_publisher(self.velocity_pub)
        self.node.destroy_publisher(self.gesture_pub)


class MultibodyDetector:
    """Class managing the holistic pose estimation process."""

    def __init__(self,
                 node: Node,
                 image_compressed: bool,
                 use_depth: bool,
                 stickman_debug: bool,
                 detection_conf_thresh: float,
                 use_cmc: bool):

        self.node = node
        self.image_compressed = image_compressed
        self.use_depth = use_depth
        self.stickman_debug = stickman_debug
        self.detection_conf_thresh = detection_conf_thresh
        self.use_cmc = use_cmc

        self.max_lost_time = BoTTrackerArgs().track_buffer / BoTTrackerArgs().frame_rate

        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=MP_MAX_NUM_PEOPLE,
            min_pose_detection_confidence=detection_conf_thresh,
            min_tracking_confidence=MP_HOL_TRACKING_CONFIDENCE)

        self.pose_detector = PoseLandmarker.create_from_options(self.options)

        # Gesture recognition
        self.gesture_options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=MP_GESTURES_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2*MP_MAX_NUM_PEOPLE)

        self.gesture_options.canned_gesture_classifier_options.category_denylist = []
        self.gesture_options.canned_gesture_classifier_options.score_threshold = 0.5

        self.gesture_recognizer = GestureRecognizer.create_from_options(self.gesture_options)

        self.processing_lock = threading.Lock()
        self.skipped_images = 0
        self.start_skipping_ts = self.node.get_clock().now()
        self.detection_start_proc_time = self.node.get_clock().now()
        self.detection_proc_duration = rclpy.duration.Duration(seconds=0.)
        self.last_frame_timestamp_ms = 0

        self.bodies = {}

        if (self.use_cmc):
            self.tracker = BoTSORT(BoTTrackerArgs(**{"cmc-method": "sparseOptFlow"}))
        else:
            self.tracker = BoTSORT(BoTTrackerArgs())

        self.ids_dict = {}

        self.ids_pub = self.node.create_publisher(
            IdsList,
            "/humans/bodies/tracked",
            1)

        self.br = CvBridge()

        image_topic = node.resolve_topic_name('image')
        self.image_subscriber = Subscriber(
            self.node,
            CompressedImage if self.image_compressed else Image,
            f'{image_topic}/compressed' if self.image_compressed else image_topic,
            qos_profile=qos_profile_sensor_data)

        if self.use_depth:
            # Here the code to detect one person only with depth information
            depth_topic = node.resolve_topic_name('depth_image')
            self.tss = ApproximateTimeSynchronizer(
                [
                    self.image_subscriber,
                    Subscriber(
                        self.node,
                        CameraInfo,
                        "camera_info",
                        qos_profile=qos_profile_sensor_data),
                    Subscriber(
                        self.node,
                        CompressedImage if self.image_compressed else Image,
                        f'{depth_topic}/compressed' if self.image_compressed else depth_topic,
                        qos_profile=qos_profile_sensor_data),
                    Subscriber(
                        self.node,
                        CameraInfo,
                        "depth_info",
                        qos_profile=qos_profile_sensor_data)
                ],
                10,
                0.1,
                allow_headerless=True
            )
            self.tss.registerCallback(self.image_callback_depth)
        else:
            self.tss = ApproximateTimeSynchronizer(
                [
                    self.image_subscriber,
                    Subscriber(
                        self.node,
                        CameraInfo,
                        "/camera_info",
                        qos_profile=qos_profile_sensor_data)
                ],
                10,
                0.2
            )
            self.tss.registerCallback(self.image_callback_rgb)

        self.node.get_logger().info("MultiBodyDetector initalized")

    def associate_hands_and_bodies(self,
                                   gesture_results,
                                   wrists_bodies,
                                   wrists_hands,
                                   wrists_ids,
                                   bodies_rois,
                                   handedness):
        if handedness != 'Left' and handedness != 'Right':
            return

        row_ind = []
        col_ind = []
        row_ind, col_ind = _find_optimal_coupling(
            list(wrists_bodies.values()), wrists_hands)
        bodies_with_hand_ids = list(wrists_bodies.keys())
        bodies_with_hand_ids = [bodies_with_hand_ids[idx]
                                for idx, _ in enumerate(bodies_with_hand_ids) if idx in row_ind]
        hands_ids = [wrists_ids[hand_id] for hand_id in col_ind]
        for idx, hand_id in enumerate(hands_ids):
            denormalized_x = int(gesture_results.hand_landmarks[hand_id][0].x*self.img_width)
            denormalized_y = int(gesture_results.hand_landmarks[hand_id][0].y*self.img_height)
            denormalized_hand_coord = np.array([denormalized_x,
                                                denormalized_y])
            if bodies_with_hand_ids[idx] in bodies_rois:
                roi_min_coords = bodies_rois[bodies_with_hand_ids[idx]][0:2]
                roi_max_coords = bodies_rois[bodies_with_hand_ids[idx]][2:]
                correction_sign = +1
                denormalized_x_min_abs = np.abs(denormalized_hand_coord[0]-roi_min_coords[0])
                denormalized_x_max_abs = np.abs(denormalized_hand_coord[0]-roi_max_coords[0])
                if denormalized_x_min_abs > denormalized_x_max_abs:
                    correction_sign = -1
                correction_term = int(correction_sign*0.01*self.img_width)
                denormalized_hand_coord[0] += correction_term
                if not (np.all(denormalized_hand_coord >= roi_min_coords) and
                        np.all(denormalized_hand_coord <= roi_max_coords)):
                    if handedness == 'Left':
                        self.bodies[bodies_with_hand_ids[idx]].left_hand = None
                    else:
                        self.bodies[bodies_with_hand_ids[idx]].right_hand = None
                    continue
            else:
                if handedness == 'Left':
                    self.bodies[bodies_with_hand_ids[idx]].left_hand = None
                else:
                    self.bodies[bodies_with_hand_ids[idx]].right_hand = None
                continue
            hand = SingleHand(gesture_results.hand_landmarks[hand_id],
                              handedness,
                              gesture_results.gestures[hand_id][0].category_name)
            if handedness == 'Left':
                self.bodies[bodies_with_hand_ids[idx]].left_hand = hand
            else:
                self.bodies[bodies_with_hand_ids[idx]].right_hand = hand

    def detect(self,
               image_rgb: Image,
               header: Header):
        """Perform the holistic pose detection."""
        if self.processing_lock.locked():
            self.skipped_images += 1
            if self.skipped_images > 100:
                self.node.get_logger().warning("hri_body_detect's mediapipe processing too slow.\
                                                Skipped 100 new incoming image\
                                                over the last %.1fsecs" %
                                               (self.node.get_clock().now()
                                                - self.start_skipping_ts))
                self.start_skipping_ts = self.node.get_clock().now()  # TODO
                self.skipped_images = 0
            return
        self.processing_lock.acquire()
        self.detection_start_proc_time = self.node.get_clock().now()  # TODO

        img_height, img_width, _ = image_rgb.shape
        self.img_height, self.img_width = img_height, img_width

        image_rgb.flags.writeable = False
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        results = None
        gesture_results = []

        # we don't use the time in the header to make sure it is monotically increasing
        # a requirement from mediapipe
        frame_timestamp_ms = int(self.node.get_clock().now().nanoseconds / 1000000)
        if frame_timestamp_ms > self.last_frame_timestamp_ms:
            self.last_frame_timestamp_ms = frame_timestamp_ms
            try:
                results = self.pose_detector.detect_for_video(
                    rgb_frame, frame_timestamp_ms)
                gesture_results = self.gesture_recognizer.recognize_for_video(
                    rgb_frame, frame_timestamp_ms)
            except RuntimeError as err:
                if 'InverseMatrixCalculator' in str(err):
                    self.node.get_logger().warning(
                        "Matrix inversion error in processing the current image,\
                        resetting mediapipe holistic")
                    self.processing_lock.release()
                    self.pose_detector = PoseLandmarker.create_from_options(
                        self.options)
                    return
                else:
                    raise
        self.processing_lock.release()

        image_rgb.flags.writeable = True
        self.image = image_rgb

        if results is None:
            return

        pose_kpt_list = []
        pose_kpt_world_list = []
        dets_array = np.empty((len(results.pose_landmarks), 6), float)
        for i in range(len(results.pose_landmarks)):

            pose_kpt = [{'x': item.x, "y": item.y, "z": item.z,
                         "visibility": item.visibility} for item in results.pose_landmarks[i]]
            pose_kpt_world = [{'x': item.x, "y": item.y, "z": item.z,
                               "visibility": item.visibility}
                              for item in results.pose_world_landmarks[i]]
            pose_kpt_list.append(pose_kpt)
            pose_kpt_world_list.append(pose_kpt_world)

            self.x_min_person = 1.0
            self.y_min_person = 1.0
            self.x_max_person = 0.
            self.y_max_person = 0.

            landmarks = [None]*32
            for j in range(0, 32):
                landmarks[j] = NormalizedPointOfInterest2D()
                landmarks[j].x = pose_kpt[j].get('x')
                landmarks[j].y = pose_kpt[j].get('y')
                landmarks[j].c = pose_kpt[j].get('visibility')

            (self.x_min_body,
                self.y_min_body,
                self.x_max_body,
                self.y_max_body) = _get_bounding_box_limits(landmarks)

            self.x_min_person = min(
                self.x_min_person,
                self.x_min_body)
            self.y_min_person = min(
                self.y_min_person,
                self.y_min_body)
            self.x_max_person = max(
                self.x_max_person,
                self.x_max_body)
            self.y_max_person = max(
                self.y_max_person,
                self.y_max_body)

            if self.x_min_person < self.x_max_person \
                    and self.y_min_person < self.y_max_person:
                self.x_min_person = max(0., self.x_min_person)
                self.y_min_person = max(0., self.y_min_person)
                self.x_max_person = min(1., self.x_max_person)
                self.y_max_person = min(1., self.y_max_person)

                dets_array[i, :] = [int(self.x_min_person*self.img_width),
                                    int(self.y_min_person*self.img_height),
                                    int(self.x_max_person*self.img_width),
                                    int(self.y_max_person*self.img_height), 1, 0]

        # Pass the detected bounding boxes to the tracker
        try:
            tracked_output = self.tracker.update(dets_array, image_rgb)
        except (ValueError, IndexError) as e:
            self.node.get_logger().error(
                "Error in the tracking process: %s. Resetting the tracker." % str(e))
            if (self.use_cmc):
                self.tracker = BoTSORT(BoTTrackerArgs(**{"cmc-method": "sparseOptFlow"}))
            else:
                self.tracker = BoTSORT(BoTTrackerArgs())
            tracked_output = self.tracker.update(dets_array, image_rgb)

        # Find a match between the track id and the body id to
        # assign the other attributes apart from the bounding box, e.g. skeleton 2D
        # (the tracker only gives a new list of bounding boxes, but with no reference
        # to the bounding boxes given as an input)
        # In the end, we need to find the assignment between the inital index in the pose
        # estimator, the id of the tracker, and the body id.

        body_ids_list = IdsList()
        body_ids_list.ids = []
        body_ids_list.header = header
        output_dict = {}
        for track in tracked_output:
            track_id = track.track_id
            if track_id not in self.ids_dict:
                body_id = "".join(random.sample(
                    "abcdefghijklmnopqrstuvwxyz", 5))

                # dictionary that relates the track_id and body_id
                self.ids_dict[track_id] = body_id

                # Create new body
                self.bodies[body_id] = SingleBody(node=self.node, use_depth=self.use_depth,
                                                  stickman_debug=self.stickman_debug,
                                                  body_id=body_id, camera_info=self.rgb_info,
                                                  img_width=self.img_width,
                                                  img_height=self.img_height,
                                                  depth_encoding=self.depth_encoding)

            output_dict[self.ids_dict[track_id]] = np.array(
                [int(b) for b in track.tlbr])
            body_ids_list.ids.append(self.ids_dict[track_id])

        # Find the closest bounding box between the tracker's output and input
        output_dict_temp = output_dict.copy()
        input_entry_to_body_id = {}
        for i in range(len(results.pose_landmarks)):
            lower_dist = 100000
            lower_id = None
            for body_id in output_dict_temp:
                dist = abs(
                    np.sum(dets_array[i, :4] - output_dict_temp[body_id]))
                if dist < lower_dist:
                    lower_dist = dist
                    lower_id = body_id
            if lower_id is not None:
                del output_dict_temp[lower_id]
                input_entry_to_body_id[i] = lower_id

        # Once we know which input of the tracker correspond to which output,
        # we process the rest of the body attributes(e.g. skeleton2d)
        # As ROI, we use the input bounding box to the tracker dets_array, but the
        # output_dict[body_id] could be chosen to use the bounding boxes from the tracker
        left_wrists_bodies = {}  # The wrists coordinates from the skeleton detection
        right_wrists_bodies = {}
        left_wrists_hands = []  # The wrists coordinates from the hand detection
        right_wrists_hands = []
        bodies_rois = {}
        for i in input_entry_to_body_id:
            body_id = input_entry_to_body_id[i]
            # if the wrist coordinates from the skeleton are not out of the frame
            # we store them to evaluate which hand they correspond to
            if ((pose_kpt_list[i][MP_LEFT_WRIST]['x'] >= 0)
               and (pose_kpt_list[i][MP_LEFT_WRIST]['x'] < 1)
               and (pose_kpt_list[i][MP_LEFT_WRIST]['y'] >= 0)
               and (pose_kpt_list[i][MP_LEFT_WRIST]['y'] < 1)):
                left_wrists_bodies[body_id] = np.array([pose_kpt_list[i][MP_LEFT_WRIST]['x'],
                                                        pose_kpt_list[i][MP_LEFT_WRIST]['y']])
            if ((pose_kpt_list[i][MP_RIGHT_WRIST]['x'] >= 0)
               and (pose_kpt_list[i][MP_RIGHT_WRIST]['x'] < 1)
               and (pose_kpt_list[i][MP_RIGHT_WRIST]['y'] >= 0)
               and (pose_kpt_list[i][MP_RIGHT_WRIST]['y'] < 1)):
                right_wrists_bodies[body_id] = np.array([pose_kpt_list[i][MP_RIGHT_WRIST]['x'],
                                                         pose_kpt_list[i][MP_RIGHT_WRIST]['y']])
            bodies_rois[body_id] = dets_array[i, :4]

        left_wrists_ids = []  # we store an ID corresponding to the detected hands
        right_wrists_ids = []
        for idx, handedness in enumerate(gesture_results.handedness):
            if handedness[0].category_name == 'Left':
                left_wrists_hands.append(np.array(
                                [gesture_results.hand_landmarks[idx][0].x,
                                 gesture_results.hand_landmarks[idx][0].y]))
                left_wrists_ids.append(idx)
            elif handedness[0].category_name == 'Right':
                right_wrists_hands.append(np.array(
                                [gesture_results.hand_landmarks[idx][0].x,
                                 gesture_results.hand_landmarks[idx][0].y]))
                right_wrists_ids.append(idx)

        if len(gesture_results.gestures) == len(gesture_results.handedness):
            if (len(left_wrists_hands) != 0) and (len(left_wrists_bodies) != 0):
                self.associate_hands_and_bodies(
                    gesture_results,
                    left_wrists_bodies,
                    left_wrists_hands,
                    left_wrists_ids,
                    bodies_rois,
                    'Left')
            else:
                for body in self.bodies:
                    self.bodies[body].left_hand = None
            if (len(right_wrists_hands) != 0) and (len(right_wrists_bodies) != 0):
                self.associate_hands_and_bodies(
                    gesture_results,
                    right_wrists_bodies,
                    right_wrists_hands,
                    right_wrists_ids,
                    bodies_rois,
                    'Right')
            else:
                for body in self.bodies:
                    self.bodies[body].right_hand = None

        self.ids_pub.publish(body_ids_list)

        for i in input_entry_to_body_id:
            body_id = input_entry_to_body_id[i]
            if body_id in self.bodies:
                self.bodies[body_id].process(
                    pose_kpt_list[i], pose_kpt_world_list[i], dets_array[i, :4],
                    header, self.rgb_info, self.depth_info, self.image_depth)

        # Remove bodies that have not been present for a while
        bodies_to_remove = []
        for body_id in self.bodies:
            if body_id not in body_ids_list.ids and header.stamp.sec \
                    - self.bodies[body_id].last_stamp.sec > self.max_lost_time:
                self.bodies[body_id].unregister()
                bodies_to_remove.append(body_id)
                if body_id in self.ids_dict.values():
                    key = list(self.ids_dict.keys())[list(self.ids_dict.values()).index(body_id)]
                    del self.ids_dict[key]
        for body in bodies_to_remove:
            if body in self.bodies:
                del self.bodies[body]

        self.detection_proc_duration = (
            self.node.get_clock().now() - self.detection_start_proc_time)

    def image_callback_depth(self,
                             rgb_img: CompressedImage | Image,
                             rgb_info: CameraInfo,
                             depth_img: CompressedImage | Image,
                             depth_info: CameraInfo):
        """Handle incoming RGB and depth images (single person)."""
        if self.image_compressed:
            rgb_img = self.br.compressed_imgmsg_to_cv2(rgb_img, desired_encoding="bgr8")
        else:
            rgb_img = self.br.imgmsg_to_cv2(rgb_img, desired_encoding="bgr8")

        if not hasattr(self, 'depth_encoding'):
            self.depth_encoding = depth_img.encoding

        if self.depth_encoding != '32FC1' and self.depth_encoding != '16UC1':
            raise ValueError('Unexpected encoding {}. '.format(self.depth_encoding) +
                             'Depth encoding should be 16UC1 or `32FC1`.')

        if self.image_compressed:
            self.image_depth = self.br.compressed_imgmsg_to_cv2(
                depth_img, desired_encoding=self.depth_encoding)
        else:
            self.image_depth = self.br.imgmsg_to_cv2(
                depth_img, desired_encoding=self.depth_encoding)
        if _builtin_time_to_secs(depth_info.header.stamp) \
                > _builtin_time_to_secs(rgb_info.header.stamp):
            header = copy.copy(depth_info.header)
            header.frame_id = rgb_info.header.frame_id  # to check
        else:
            header = copy.copy(rgb_info.header)
        self.depth_info = depth_info
        self.rgb_info = rgb_info
        self.detect(rgb_img, header)

    def image_callback_rgb(self,
                           rgb_img: CompressedImage | Image,
                           rgb_info: CameraInfo):
        """Handle incoming RGB images."""
        if self.image_compressed:
            rgb_img = self.br.compressed_imgmsg_to_cv2(rgb_img, desired_encoding="bgr8")
        else:
            rgb_img = self.br.imgmsg_to_cv2(rgb_img, desired_encoding="bgr8")
        header = copy.copy(rgb_info.header)
        self.rgb_info = rgb_info
        self.depth_encoding = None
        self.depth_info = None
        self.image_depth = None
        self.detect(rgb_img, header)

    def get_image_topic(self):
        """Return the image stream topic used to perform pose detection."""
        return self.image_subscriber.topic_name

    def check_timeout(self) -> bool:
        """
        Perform timeout check.

        Returns whether the time from the last detection has exceeded
        the timeout threshold (given that the processing lock
        is actually locked).
        """
        return ((self.node.get_clock().now()
                 - self.detection_start_proc_time).nanoseconds/1e9
                > BODY_DETECTION_PROC_TIMEOUT_ERROR
                and self.processing_lock.locked())

    def get_proc_time(self) -> rclpy.duration.Duration:
        """Return the last estimation process time."""
        return self.detection_proc_duration
