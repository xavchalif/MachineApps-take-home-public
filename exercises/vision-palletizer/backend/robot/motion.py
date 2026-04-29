"""
Motion Controller
================

Implements robot motion commands for pick and place operations.
"""

from typing import Optional
import numpy as np

from .connection import RobotConnection


class MotionController:
    """
    Controls robot motion for palletizing operations.

    Coordinates:
    - All positions are in meters
    - All orientations are in radians (axis-angle representation for UR)
    """

    APPROACH_HEIGHT_OFFSET = 0.100  # 100mm, safely above the 50mm requirement
    DEFAULT_VELOCITY = 0.5
    DEFAULT_ACCELERATION = 0.5
    HOME_JOINTS = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    WORKSPACE_LIMITS = {
        "x": (-0.85, 0.85),
        "y": (-0.85, 0.85),
        "z": (0.02, 1.20),
    }

    def __init__(self, connection: RobotConnection):
        """
        Initialize motion controller.
        
        Args:
            connection: Active robot connection instance.
        """
        self.connection = connection
        self._gripper_closed = False

    def move_to_home(self) -> bool:
        """
        Move robot to home/safe position.
        
        Returns:
            True if move completed successfully.
        """
        return self._move_joint(self.HOME_JOINTS)

    def move_to_pick(
        self,
        position: list[float],
        orientation: Optional[list[float]] = None,
    ) -> bool:
        """
        Execute pick motion sequence.
        
        Args:
            position: [x, y, z] pick position in robot base frame (meters)
            orientation: [rx, ry, rz] tool orientation (axis-angle, radians)
                        If None, use default downward orientation.
        
        Returns:
            True if pick completed successfully.
        """
        target = self._build_pose(position, orientation)
        approach = target.copy()
        approach[2] += self.APPROACH_HEIGHT_OFFSET

        self._validate_pose(target)
        self._validate_pose(approach)

        approach_joints = self._inverse_kinematics(approach)
        return (
            self._move_joint(approach_joints)
            and self._move_linear(target)
            and self.close_gripper()
            and self._move_linear(approach)
        )

    def move_to_place(
        self,
        position: list[float],
        orientation: Optional[list[float]] = None,
    ) -> bool:
        """
        Execute place motion sequence.
        
        Args:
            position: [x, y, z] place position in robot base frame (meters)
            orientation: [rx, ry, rz] tool orientation (axis-angle, radians)
                        If None, use default downward orientation.
        
        Returns:
            True if place completed successfully.
        """
        target = self._build_pose(position, orientation)
        approach = target.copy()
        approach[2] += self.APPROACH_HEIGHT_OFFSET

        self._validate_pose(target)
        self._validate_pose(approach)
        return (
            self._move_linear(approach)
            and self._move_linear(target)
            and self.open_gripper()
            and self._move_linear(approach)
        )

    def open_gripper(self) -> bool:
        """
        Open the gripper to release object.
        
        Returns:
            True if gripper opened successfully.
        """
        self._gripper_closed = False
        print("[MOCK] Gripper opened")
        return True

    def close_gripper(self) -> bool:
        """
        Close the gripper to grasp object.
        
        Returns:
            True if gripper closed successfully.
        """
        self._gripper_closed = True
        print("[MOCK] Gripper closed")
        return True

    def _move_linear(
        self,
        pose: list[float],
        velocity: float = DEFAULT_VELOCITY,
        acceleration: float = DEFAULT_ACCELERATION,
    ) -> bool:
        """
        Execute linear move to target pose.
        
        Args:
            pose: [x, y, z, rx, ry, rz] target pose
            velocity: Move velocity in m/s
            acceleration: Move acceleration in m/s²
        
        Returns:
            True if move completed.
        """
        if self.connection.is_mock_mode():
            print(f"[MOCK] moveL to {pose[:3]}")
            return True
        if not self.connection.ensure_connected() or self.connection.control is None:
            raise RuntimeError("Robot not connected")
        return bool(self.connection.control.moveL(pose, velocity, acceleration))

    def _move_joint(
        self,
        joints: list[float],
        velocity: float = 1.0,
        acceleration: float = 1.0,
    ) -> bool:
        """
        Execute joint move to target configuration.
        
        Args:
            joints: List of 6 joint angles in radians
            velocity: Joint velocity in rad/s
            acceleration: Joint acceleration in rad/s²
        
        Returns:
            True if move completed.
        """
        if self.connection.is_mock_mode():
            print(f"[MOCK] moveJ to {joints}")
            return True
        if not self.connection.ensure_connected() or self.connection.control is None:
            raise RuntimeError("Robot not connected")
        return bool(self.connection.control.moveJ(joints, velocity, acceleration))

    def get_default_orientation(self) -> list[float]:
        """
        Get default tool orientation for picking (pointing down).
        
        Returns:
            [rx, ry, rz] in axis-angle representation.
        
        Note: For a tool pointing straight down (Z toward floor),
        the rotation from base frame is typically [0, π, 0] or [π, 0, 0]
        depending on your tool frame setup.
        """
        return [0.0, np.pi, 0.0]
    
    def orientation_from_yaw(self, yaw_rad: float) -> list[float]:
        """Rotate the downward tool around Z for detected box yaw compensation."""
        return [0.0, np.pi, yaw_rad]

    def _build_pose(self, position: list[float], orientation: Optional[list[float]]) -> list[float]:
        if len(position) != 3:
            raise ValueError("position must be [x, y, z]")
        tool_orientation = orientation or self.get_default_orientation()
        if len(tool_orientation) != 3:
            raise ValueError("orientation must be [rx, ry, rz]")
        return [float(v) for v in position] + [float(v) for v in tool_orientation]

    def _validate_pose(self, pose: list[float]) -> None:
        x, y, z = pose[:3]
        limits = self.WORKSPACE_LIMITS
        if not (limits["x"][0] <= x <= limits["x"][1]):
            raise ValueError(f"x={x:.3f}m is outside workspace limits {limits['x']}")
        if not (limits["y"][0] <= y <= limits["y"][1]):
            raise ValueError(f"y={y:.3f}m is outside workspace limits {limits['y']}")
        if not (limits["z"][0] <= z <= limits["z"][1]):
            raise ValueError(f"z={z:.3f}m is outside workspace limits {limits['z']}")
        
    def _inverse_kinematics(self, pose: list[float]) -> list[float]:
        """Convert a Cartesian TCP pose [x, y, z, rx, ry, rz] to robot joint angles."""
        if self.connection.is_mock_mode():
            print(f"[MOCK] IK for pose {pose}")
            return self.HOME_JOINTS.copy()

        if not self.connection.ensure_connected() or self.connection.control is None:
            raise RuntimeError("Robot not connected")

        joints = self.connection.control.getInverseKinematics(pose)

        if joints is None or len(joints) != 6:
            raise RuntimeError(f"Inverse kinematics failed for pose: {pose}")

        return [float(joint) for joint in joints]
