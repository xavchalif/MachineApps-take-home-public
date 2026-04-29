"""
Coordinate Transformations
=========================

Transform coordinates between camera frame and robot base frame.
"""

import numpy as np

CAMERA_TRANSLATION_MM = np.array([500.0, 300.0, 500.0], dtype=float)
CAMERA_ROLL_RAD = np.deg2rad(15.0)
CAMERA_PITCH_RAD = np.deg2rad(-10.0)
CAMERA_YAW_RAD = np.deg2rad(45.0)


def build_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 3x3 rotation matrix using intrinsic Z -> Y -> X rotations."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def camera_to_robot(point_camera: np.ndarray) -> np.ndarray:
    """Transform [x, y, z] from camera frame to robot base frame, in mm."""
    point = np.asarray(point_camera, dtype=float).reshape(3)
    rotation = build_rotation_matrix(CAMERA_ROLL_RAD, CAMERA_PITCH_RAD, CAMERA_YAW_RAD)
    return rotation @ point + CAMERA_TRANSLATION_MM


def robot_to_camera(point_robot: np.ndarray) -> np.ndarray:
    """Transform [x, y, z] from robot base frame to camera frame, in mm."""
    point = np.asarray(point_robot, dtype=float).reshape(3)
    rotation = build_rotation_matrix(CAMERA_ROLL_RAD, CAMERA_PITCH_RAD, CAMERA_YAW_RAD)
    return rotation.T @ (point - CAMERA_TRANSLATION_MM)


def build_homogeneous_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transformation matrix."""
    rot = np.asarray(rotation, dtype=float)
    trans = np.asarray(translation, dtype=float).reshape(3)
    if rot.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return transform
