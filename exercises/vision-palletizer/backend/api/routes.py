"""
Palletizer API Routes
====================

FastAPI routes for palletizer control.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np

from palletizer.grid import calculate_place_positions
from palletizer.state_machine import PalletizerStateMachine
from robot.connection import RobotConnection
from robot.motion import MotionController
from transforms.coordinate import camera_to_robot

router = APIRouter()

_robot_connection = RobotConnection()
_motion_controller = MotionController(_robot_connection)
palletizer = PalletizerStateMachine(motion_controller=_motion_controller)


class PalletConfig(BaseModel):
    """Configuration for palletizing operation."""

    rows: int = Field(..., ge=1, le=10, description="Number of rows in the grid")
    cols: int = Field(..., ge=1, le=10, description="Number of columns in the grid")
    box_width_mm: float = Field(..., gt=0, description="Box width in mm (X direction)")
    box_depth_mm: float = Field(..., gt=0, description="Box depth in mm (Y direction)")
    box_height_mm: float = Field(..., gt=0, description="Box height in mm (Z direction)")
    pallet_origin_x_mm: float = Field(..., description="Pallet origin X in mm")
    pallet_origin_y_mm: float = Field(..., description="Pallet origin Y in mm")
    pallet_origin_z_mm: float = Field(..., description="Pallet origin Z in mm")
    spacing_mm: float = Field(10.0, ge=0, description="Gap between boxes in mm")

    class Config:
        json_schema_extra = {
            "example": {
                "rows": 2,
                "cols": 2,
                "box_width_mm": 100.0,
                "box_depth_mm": 100.0,
                "box_height_mm": 50.0,
                "pallet_origin_x_mm": 400.0,
                "pallet_origin_y_mm": -200.0,
                "pallet_origin_z_mm": 100.0,
                "spacing_mm": 10.0,
            }
        }


class VisionDetection(BaseModel):
    """Simulated vision detection of a box."""

    x_mm: float = Field(..., description="Box X position in camera frame (mm)")
    y_mm: float = Field(..., description="Box Y position in camera frame (mm)")
    z_mm: float = Field(..., description="Box Z position in camera frame (mm)")
    yaw_deg: Optional[float] = Field(0.0, description="Box rotation about Z (degrees)")

    class Config:
        json_schema_extra = {
            "example": {
                "x_mm": 50.0,
                "y_mm": -30.0,
                "z_mm": 0.0,
                "yaw_deg": 15.0,
            }
        }


class StatusResponse(BaseModel):
    state: str = Field(..., description="Current state machine state")
    current_box: int = Field(..., description="Current box index (0-based)")
    total_boxes: int = Field(..., description="Total boxes to palletize")
    error: Optional[str] = Field(None, description="Error message if in FAULT state")


class ConfigResponse(BaseModel):
    success: bool
    message: str
    grid_size: Optional[str] = None


class CommandResponse(BaseModel):
    success: bool
    message: str


@router.post("/configure", response_model=ConfigResponse)
async def configure_palletizer(config: PalletConfig):
    """Configure grid dimensions, box size, spacing, and pallet origin."""
    ok = palletizer.configure(
        rows=config.rows,
        cols=config.cols,
        box_size_mm=(config.box_width_mm, config.box_depth_mm, config.box_height_mm),
        pallet_origin_mm=(
            config.pallet_origin_x_mm,
            config.pallet_origin_y_mm,
            config.pallet_origin_z_mm,
        ),
        spacing_mm=config.spacing_mm,
    )
    if not ok:
        raise HTTPException(status_code=409, detail=palletizer.context.error_message or "Palletizer must be IDLE to configure")
    return ConfigResponse(success=True, message="Palletizer configured", grid_size=f"{config.rows}x{config.cols}")


@router.post("/start", response_model=CommandResponse)
async def start_palletizer():
    """Start the palletizing sequence."""
    ok = palletizer.begin()
    if not ok:
        raise HTTPException(status_code=409, detail=palletizer.context.error_message or "Unable to start palletizer")
    return CommandResponse(success=True, message="Palletizing sequence completed")


@router.post("/stop", response_model=CommandResponse)
async def stop_palletizer():
    """Gracefully stop operation and return to IDLE."""
    ok = palletizer.stop()
    if not ok:
        raise HTTPException(status_code=409, detail="Unable to stop from current state")
    return CommandResponse(success=True, message="Palletizer stopped")


@router.post("/reset", response_model=CommandResponse)
async def reset_palletizer():
    """Reset from FAULT state."""
    ok = palletizer.reset()
    if not ok:
        raise HTTPException(status_code=409, detail="Unable to reset from current state")
    return CommandResponse(success=True, message="Palletizer reset")


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Return current state, progress, and any error."""
    return StatusResponse(**palletizer.progress)


@router.post("/vision/detect", response_model=CommandResponse)
async def simulate_vision_detection(detection: VisionDetection):
    """Add a camera-frame detection to the queue for the next palletizing run."""
    palletizer.add_detection(detection.x_mm, detection.y_mm, detection.z_mm, detection.yaw_deg or 0.0)
    robot_mm = camera_to_robot(np.array([detection.x_mm, detection.y_mm, detection.z_mm]))
    return CommandResponse(
        success=True,
        message=f"Detection queued; robot-frame pick target is {[round(float(v), 3) for v in robot_mm]} mm",
    )


@router.get("/debug/positions")
async def get_calculated_positions():
    """Return calculated place positions for debugging."""
    if not palletizer.context.place_positions:
        palletizer.context.place_positions = calculate_place_positions(
            palletizer.context.rows,
            palletizer.context.cols,
            palletizer.context.box_size_mm,
            palletizer.context.pallet_origin_mm,
            palletizer.context.spacing_mm,
        )
    return {"positions_mm": palletizer.context.place_positions}


@router.post("/debug/transform")
async def test_transform(detection: VisionDetection):
    """Transform a camera-frame point to robot base frame for verification."""
    camera_mm = np.array([detection.x_mm, detection.y_mm, detection.z_mm], dtype=float)
    robot_mm = camera_to_robot(camera_mm)
    return {
        "camera_frame_mm": camera_mm.tolist(),
        "robot_base_frame_mm": [round(float(v), 6) for v in robot_mm],
        "yaw_deg": detection.yaw_deg or 0.0,
    }
