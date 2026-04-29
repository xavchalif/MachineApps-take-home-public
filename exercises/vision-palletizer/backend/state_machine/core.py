"""Small compatibility shim for the vention-state-machine API used here."""
from enum import Enum
from typing import Any


class BaseTriggers(Enum):
    TO_FAULT = "to_fault"
    RESET = "reset"


class StateMachine:
    def __init__(self, states: Any = None, transitions: list | None = None, enable_last_state_recovery: bool = False):
        self.states = states
        self.transitions = transitions or []
        self.enable_last_state_recovery = enable_last_state_recovery
        self.state = "ready"
        self._enter_callbacks = {}
        self._change_callbacks = []
        self._collect_callbacks()

    def _collect_callbacks(self) -> None:
        # Inspect class attributes so @property descriptors are not evaluated
        # before subclasses finish initializing their context.
        for cls in reversed(type(self).mro()):
            for name, raw_attr in cls.__dict__.items():
                enter_state = getattr(raw_attr, "_on_enter_state", None)
                if enter_state:
                    self._enter_callbacks[enter_state] = getattr(self, name)
                if getattr(raw_attr, "_on_state_change", False):
                    self._change_callbacks.append(getattr(self, name))

    def trigger(self, trigger: str) -> None:
        old_state = self.state
        if trigger == BaseTriggers.TO_FAULT.value:
            self.state = "fault"
        elif trigger == BaseTriggers.RESET.value and self.state == "fault":
            self.state = "ready"
        else:
            for transition in self.transitions:
                if transition.trigger == trigger and self._state_name(transition.source) == self.state:
                    self.state = self._state_name(transition.target)
                    break
            else:
                raise RuntimeError(f"No transition for trigger {trigger!r} from state {self.state!r}")

        for callback in self._change_callbacks:
            callback(old_state, self.state, trigger)
        callback = self._enter_callbacks.get(self.state)
        if callback:
            callback(None)

    @staticmethod
    def _state_name(state: Any) -> str:
        return state if isinstance(state, str) else str(state)
