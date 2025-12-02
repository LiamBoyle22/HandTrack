from enum import Enum, auto

class ControlState(Enum):
    
    IDLE = auto()
    ACTIVE = auto()

class GestureStateMachine:

    def __init__(self):
        self.state = ControlState.IDLE

    def progress(self) -> float:
        return 1.0 if self.state == ControlState.ACTIVE else 0.0

    def update(self, gesture_label: str) -> ControlState:
        if gesture_label == "ThumbsUp":
            self.state = ControlState.ACTIVE
        elif gesture_label == "ThumbsDown":
            self.state = ControlState.IDLE

        return self.state