hri_body_detect
=======================

![skeleton detection](doc/skeleton_detection.png)

Overview
--------

> :warning: some of the links are yet to be updated and
> may be pointing to the original ROS page. As soon
> as all the involved components will be officially
> documented for ROS 2 as well, we will update
> this document.


`hri_body_detect` is a [ROS4HRI](https://wiki.ros.org/hri)-compatible
2D and 3D body pose estimation node.

It is built on top of [Google Mediapipe 3D body pose estimation](https://google.github.io/mediapipe/solutions/pose.html).

The node provides the 2D and 3D pose estimation for the detected humans 
in the scene, implementing a robust solution to self-occlusions.

This node performs the body-pose detection pipeline, 
publishing information under the ROS4HRI naming convention regarding
the body ids (on the `/humans/bodies/tracked` topic), the bodies bounding box,
and the jointstate of the bodys' skeleton.

To estimate the body position, the node does not need a RGB-D camera,
only RGB is required. However, using RGB-D camera provides a more 
accurate depth estimation.

**Important**: to estimate the body depth without using a depth sensor, 
a calibrated RGB camera is required. 
You can follow [this tutorial](https://navigation.ros.org/tutorials/docs/camera_calibration.html)
to properly calibrate your camera.

Launch
------

The launch file `hri_body_detect.launch.py` is intended to be used in PAL robots following PAPS-007
For general usage, use `hri_body_detect_with_args.launch.py`:

`ros2 launch hri_body_detect hri_body_detect_with_args.launch.py <parameters>`

ROS API
-------

### Parameters

#### Node parameters:

- `image_compressed` (default, `True`): selects the compressed image transport
- `use_depth` (default: `False`): whether or not to rely on depth images 
  for estimating body movement in the scene. When this is `False`, the node
  estimates the body position in the scene solving a P6P problem for the
  face and approximating the position from this, using pinhole camera
  model geometry. 
- `stickman_debug` (default: `False`): whether or not to publish frames
  representing the body skeleton directly using the raw results from mediapipe
  3D body pose estimation. These debug frames are *not* oriented to align 
  with the body links (ie, only the 3D location of the frame is useful).
- `detection_conf_thresh` (default: `0.5`): threshold to apply to the mediapipe
  pose detection. Higher thresholds will lead to less detected bodies, but also
  less false positives.
- `use_cmc` (default: `False`): whether or not to enable camera motion
  compensation in the tracker. It compensates the movement of the camera respect
  to the world during tracking, but it is CPU intensive as it is computing the
  optical flow.

#### hri_body_detect_with_args.launch parameters:

- `use_depth` (default: `False`): equivalent to `use_depth` node parameter.
- `stickman_debug` (default: `False`): equivalent to `stickman_debug` node parameter.
- `detection_conf_thresh` (default: `0.5`): equivalent to `detection_conf_thresh` node parameter.
- `use_cmc` (default: `False`): equivalent to `use_cmc` node parameter.
- `rgb_camera` (default: ` `): rgb camera topics namespace.
- `rgb_camera_topic` (default: `$(arg rgb_camera)/image_raw`): rgb camera
  raw image topic. 
- `rgb_camera_info` (default: `$(arg rgb_camera)/camera_info`): rgb camera
  info topic.
- `depth_camera` (default: ` `): depth camera topics namespace. 
- `depth_camera_topic` (default: `$(arg depth_camera)/image_rect_raw`): depth 
  camera rectified raw image topic.
- `depth_camera_info` (default: `$(arg depth_camera)/camera_info`): depth 
  camera info topic.

### Topics

`hri_body_detect` follows the ROS4HRI conventions (REP-155). In particular, 
refer to the REP to know the list and position of the 2D/3D skeleton 
points published by the node.

#### Subscribed topics

- `camera_info`
  ([sensor_msgs/CameraInfo](https://docs.ros2.org/latest/api/sensor_msgs/msg/CameraInfo.html)):
  rgb camera meta information
- `image`
  ([sensor_msgs/Image](https://docs.ros2.org/latest/api/sensor_msgs/msg/Image.html)):
  only if `image_compressed` is false;
  rgb image, processed for body detection and 3D body pose estimation.
- `image/compressed`
  ([sensor_msgs/CompressedImage](https://docs.ros2.org/latest/api/sensor_msgs/msg/CompressedImage.html)):
  only if `image_compressed` is true;
  rgb image, processed for body detection and 3D body pose estimation;
  note that the suffix `/compressed` is added *after* the remapping is resolved,
  so you should remap only `image` regardless of the `image_compressed` value.
- `depth_info`
  ([sensor_msgs/CameraInfo](https://docs.ros2.org/latest/api/sensor_msgs/msg/CameraInfo.html)):
  depth camera meta information
- `depth_image/compressed`
  ([sensor_msgs/CompressedImage](https://docs.ros2.org/latest/api/sensor_msgs/msg/CompressedImage.html)):
  only if `image_compressed` is true;
  depth image used to estimate the 3D body position with respect to the camera;
  note that the suffix `/compressed` is added *after* the remapping is resolved,
  so you should remap only `depth_image` regardless of the `image_compressed` value.

#### Published topics

- `/humans/bodies/tracked`
  ([hri_msgs/IdsList](http://docs.ros.org/en/api/hri_msgs/html/msg/IdsList.html)):
  list of the bodies currently detected. There will be only
  one body in the list.
- `/humans/bodies/<body_id>/skeleton2d`
  ([hri_msgs/Skeleton2D](http://docs.ros.org/en/api/hri_msgs/html/msg/Skeleton2D.html)):
  detected 2D skeleton points.
- `/humans/bodies/<body_id>/joint_states`
  ([sensor_msgs/JointState](https://docs.ros2.org/latest/api/sensor_msgs/msg/JointState.html)):
  skeleton joints state.
- `/humans/bodies/<body_id>/position`:
  ([geometry_msgs/PointStamped](https://docs.ros2.org/latest/api/geometry_msgs/msg/PointStamped.html)):
  filtered body position, representing the point between the hips of the tracked body. Only published 
  when `use_depth = True`.
- `/humans/bodies/<body_id>/velocity`:
  ([geometry_msgs/TwistStamped](https://docs.ros2.org/latest/api/geometry_msgs/msg/TwistStamped.html)):
  filtered body velocity. Only published when `use_depth = True`.
- `/humans/bodies/<body_id>/roi`
  ([hri_msgs/NormalizedRegionOfInterest2D](http://docs.ros.org/en/noetic/api/hri_msgs/html/msg/NormalizedRegionOfInterest2D.html)):
  body bounding box in normalized image coordinates.
- `/humans/bodies/<body_id>/urdf`
  ([std_msgs/String](https://docs.ros2.org/latest/api/std_msgs/msg/String.html)):
  body URDF.

Visualization
-------------

It is possible to visualize the results of the body pose estimation 
in rviz using the [hri_rviz](https://github.com/ros4hri/hri_rviz) 
Skeleton plugin. A visualization example is reported in the image above. 









