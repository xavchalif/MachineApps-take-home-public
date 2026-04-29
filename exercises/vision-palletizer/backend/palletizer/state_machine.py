"""
Palletizer State Machine

Manages the lifecycle of palletizing operations using vention-state-machine.
"""

from enum import Enum, auto
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import math

import numpy as np
from state_machine.core import StateMachine, BaseTriggers
from state_machine.defs import StateGroup, State, Trigger
from state_machine.decorators import on_enter_state, on_state_change

from palletizer.grid import calculate_place_positions
from transforms.coordinate import camera_to_robot


class PalletizerState(Enum):
    IDLE = auto()
    HOMING = auto()
    PICKING = auto()
    PLACING = auto()
    FAULT = auto()


class Running(StateGroup):
    homing: State = State()
    picking: State = State()
    placing: State = State()


class States:
    running = Running()


class Triggers:
    finished_homing = Trigger("finished_homing")
    finished_picking = Trigger("finished_picking")
    finished_placing = Trigger("finished_placing")
    cycle_complete = Trigger("cycle_complete")
    stop = Trigger("stop")


TRANSITIONS = [
    Trigger("start").transition("ready", States.running.homing),
    Triggers.finished_homing.transition(States.running.homing, States.running.picking),
    Triggers.finished_picking.transition(States.running.picking, States.running.placing),
    Triggers.finished_placing.transition(States.running.placing, States.running.picking),
    Triggers.cycle_complete.transition(States.running.placing, "ready"),
    Triggers.stop.transition(States.running.homing, "ready"),
    Triggers.stop.transition(States.running.picking, "ready"),
    Triggers.stop.transition(States.running.placing, "ready"),
]


@dataclass
class PalletizerContext:
    rows: int = 2
    cols: int = 2
    box_size_mm: tuple[float, float, float] = (100.0, 100.0, 50.0)
    pallet_origin_mm: tuple[float, float, float] = (400.0, -200.0, 100.0)
    spacing_mm: float = 10.0
    current_box_index: int = 0
    total_boxes: int = 0
    pick_position: Optional[tuple[float, float, float]] = None
    pick_yaw_deg: float = 0.0
    place_positions: list[tuple[float, float, float]] = field(default_factory=list)
    detections: list[dict] = field(default_factory=list)
    error_message: str = ""


class PalletizerStateMachine(StateMachine):
    """State machine for palletizing operations."""

    def __init__(self, motion_controller=None, detections_path: Optional[str] = None):
        super().__init__(
            states=States,
            transitions=TRANSITIONS,
            enable_last_state_recovery=False,
        )
        self.context = PalletizerContext()
        self.motion_controller = motion_controller
        self.detections_path = Path(detections_path) if detections_path else Path(__file__).resolve().parents[1] / "data" / "camera_detections.json"

    @property
    def current_state(self) -> PalletizerState:
        mapping = {
            "ready": PalletizerState.IDLE,
            "fault": PalletizerState.FAULT,
            "Running_homing": PalletizerState.HOMING,
            "Running_picking": PalletizerState.PICKING,
            "Running_placing": PalletizerState.PLACING,
        }
        return mapping.get(self.state, PalletizerState.IDLE)

    @property
    def progress(self) -> dict:
        return {
            "state": self.current_state.name,
            "current_box": self.context.current_box_index,
            "total_boxes": self.context.total_boxes,
            "error": self.context.error_message or None,
        }

    def configure(
        self,
        rows: int,
        cols: int,
        box_size_mm: tuple[float, float, float],
        pallet_origin_mm: tuple[float, float, float],
        spacing_mm: float = 10.0,
    ) -> bool:
        if self.current_state != PalletizerState.IDLE:
            return False
        try:
            self.context.place_positions = calculate_place_positions(
                rows, cols, box_size_mm, pallet_origin_mm, spacing_mm
            )
        except Exception as exc:
            self.context.error_message = str(exc)
            return False

        self.context.rows = rows
        self.context.cols = cols
        self.context.box_size_mm = box_size_mm
        self.context.pallet_origin_mm = pallet_origin_mm
        self.context.spacing_mm = spacing_mm
        self.context.total_boxes = rows * cols
        self.context.current_box_index = 0
        self.context.error_message = ""
        return True

    def add_detection(self, x_mm: float, y_mm: float, z_mm: float, yaw_deg: float = 0.0) -> None:
        self.context.detections.append({
            "x_mm": x_mm,
            "y_mm": y_mm,
            "z_mm": z_mm,
            "yaw_deg": yaw_deg,
        })

    def begin(self) -> bool:
        if self.current_state != PalletizerState.IDLE:
            return False
        if not self.context.place_positions:
            self.configure(
                self.context.rows,
                self.context.cols,
                self.context.box_size_mm,
                self.context.pallet_origin_mm,
                self.context.spacing_mm,
            )
        if not self.context.detections:
            self._load_detections_from_file()
        self.context.total_boxes = min(len(self.context.place_positions), len(self.context.detections))
        if self.context.total_boxes <= 0:
            self.fault("No detections or place positions available")
            return False
        self.context.current_box_index = 0
        try:
            self.trigger("start")
            return True
        except Exception as exc:
            self.fault(str(exc))
            return False

    def stop(self) -> bool:
        if self.current_state == PalletizerState.IDLE:
            return True
        try:
            self.trigger("stop")
            return True
        except Exception:
            return False

    def reset(self) -> bool:
        try:
            self.trigger(BaseTriggers.RESET.value)
            self.context.error_message = ""
            self.context.current_box_index = 0
            return True
        except Exception:
            return False

    def fault(self, message: str) -> bool:
        self.context.error_message = message
        try:
            self.trigger(BaseTriggers.TO_FAULT.value)
            return True
        except Exception:
            return False

    @on_enter_state(States.running.homing)
    def on_enter_homing(self, _):
        try:
            if self.motion_controller:
                if not self.motion_controller.move_to_home():
                    raise RuntimeError("Robot failed to move home")
            self.trigger("finished_homing")
        except Exception as exc:
            self.fault(str(exc))

    @on_enter_state(States.running.picking)
    def on_enter_picking(self, _):
        try:
            if self.context.current_box_index >= self.context.total_boxes:
                self.trigger("cycle_complete")
                return
            detection = self.context.detections[self.context.current_box_index]
            robot_mm = camera_to_robot(np.array([
                detection["x_mm"], detection["y_mm"], detection["z_mm"]
            ], dtype=float))
            self.context.pick_position = tuple(float(v) for v in robot_mm)
            self.context.pick_yaw_deg = float(detection.get("yaw_deg", 0.0) or 0.0)

            if self.motion_controller:
                orientation = self.motion_controller.orientation_from_yaw(math.radians(self.context.pick_yaw_deg))
                if not self.motion_controller.move_to_pick((robot_mm / 1000.0).tolist(), orientation):
                    raise RuntimeError("Pick motion failed")
            self.trigger("finished_picking")
        except Exception as exc:
            self.fault(str(exc))

    @on_enter_state(States.running.placing)
    def on_enter_placing(self, _):
        try:
            place_mm = self.context.place_positions[self.context.current_box_index]
            if self.motion_controller:
                orientation = self.motion_controller.orientation_from_yaw(math.radians(self.context.pick_yaw_deg))
                place_m = [float(v) / 1000.0 for v in place_mm]
                if not self.motion_controller.move_to_place(place_m, orientation):
                    raise RuntimeError("Place motion failed")

            self.context.current_box_index += 1
            if self.context.current_box_index >= self.context.total_boxes:
                self.trigger("cycle_complete")
            else:
                self.trigger("finished_placing")
        except Exception as exc:
            self.fault(str(exc))

    @on_state_change
    def on_any_state_change(self, old_state: str, new_state: str, trigger: str):
        print(f"[STATE] {old_state} -> {new_state} via {trigger}")

    def _load_detections_from_file(self) -> None:
        with self.detections_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        detections = payload.get("detections", [])
        self.context.detections = [
            {
                "x_mm": float(d["x_mm"]),
                "y_mm": float(d["y_mm"]),
                "z_mm": float(d.get("z_mm", 0.0)),
                "yaw_deg": float(d.get("yaw_deg", 0.0) or 0.0),
            }
            for d in detections
        ]
