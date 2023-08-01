ros2 topic pub /tf tf2_msgs/TFMessage "
transforms:
- header:
    stamp: {sec: 0, nanosec: 0}
    frame_id: 'camera_depth_optical_frame'
  child_frame_id: 'calibration_box'
  transform:
    translation: {x: -0.29017985, y: -0.0312448, z: 1.1571045}
    rotation: {x: -0.49639403, y: 0.61554146, z: -0.06636464, w: 0.60852067}
" -r 1
