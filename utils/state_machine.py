# utils/state_machine.py
from enum import Enum, auto
import time

class ControlState(Enum):
    IDLE = auto()
    ARMING = auto()
    ACTIVE = auto()

class GestureStateMachine:
    def __init__(self, hold_seconds: float = 3.0, drop_grace: float = 0.35):
        self.state = ControlState.IDLE
        self._arming_start = None
        self.hold_seconds = hold_seconds
        self.drop_grace = drop_grace  # allows brief landmark/label hiccups
        self._last_fist_time = None

    def progress(self) -> float:
        if self.state == ControlState.ARMING and self._arming_start is not None:
            return max(0.0, min(1.0, (time.monotonic() -self._arming_start) / self.hold_seconds))
        return 1.0 if self.state == ControlState.ACTIVE else 0.0

    def update(self, gesture_label: str) -> ControlState:
        now = time.monotonic()
        is_fist = (gesture_label == "fist")
        is_five = (gesture_label == "five")

        if self.state == ControlState.IDLE:
            if is_fist:
                self.state = ControlState.ARMING
                self._arming_start = now
        elif self.state == ControlState.ARMING:
            if is_fist:
                if now - self._arming_start >= self.hold_seconds:
                    self.state = ControlState.ACTIVE
            else:
                # brief drops tolerated
                if now - self._arming_start > self.drop_grace:
                    self.state = ControlState.IDLE
                    self._arming_start = None
        elif self.state == ControlState.ACTIVE:
            # Optional: make another 3s fist hold disable control
            if is_fist:
                if self._last_fist_time is None:
                    self._last_fist_time = now
                elif now - self._last_fist_time >= self.hold_seconds:
                    self.state = ControlState.IDLE
                    self._last_fist_time = None
            else:
                # reset the disable timer if hand changes
                self._last_fist_time = None

        return self.state
