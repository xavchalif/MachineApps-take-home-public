"""Small compatibility shim for the vention-state-machine API used here."""
from dataclasses import dataclass
from typing import Any


class State:
    def __set_name__(self, owner: type, name: str) -> None:
        self.name = f"{owner.__name__}_{name}"

    def __str__(self) -> str:
        return self.name


class StateGroup:
    pass


@dataclass(frozen=True)
class Transition:
    trigger: str
    source: Any
    target: Any


class Trigger:
    def __init__(self, name: str):
        self.name = name

    def transition(self, source: Any, target: Any) -> Transition:
        return Transition(self.name, source, target)
