from __future__ import annotations
import cv2
import time
from typing import Optional, Tuple

from actions.action_mapper import ActionMapper
from gesture_rec.hand_detect import HandDetector
from gesture_rec.gesture_class import GestureClassifier
from gesture_rec import gesture_config as config
from utils.state_machine import GestureStateMachine, ControlState

#Fix: disarm animation still does not show


#Arm after 3s fist, allow brief hiccups during the hold
state_machine = GestureStateMachine(hold_seconds=3.0, drop_grace=0.6)

def smooth(prev: Optional[Tuple[int, int]], curr: Tuple[int, int], alpha: float = 0.7) -> Tuple[int, int]:
    """EMA smoothing for cursor coordinates (alpha = weight of previous)."""
    if prev is None:
        return curr
    return (
        int(alpha * prev[0] + (1 - alpha) * curr[0]),
        int(alpha * prev[1] + (1 - alpha) * curr[1]),
    )

def _normalize_for_state(gesture: str) -> str:
    """Map your classifier constants to simple labels the state machine expects."""
    if gesture == GestureClassifier.GESTURE_FIST:
        return "fist"
    if gesture == GestureClassifier.GESTURE_FIVE_FINGERS:
        return "five"
    return "other"

class App:
    def __init__(self):
        #Hand detector + classifier
        self.detector = HandDetector(
            max_num_hands = config.MAX_NUM_HANDS,
            detection_confidence = config.HAND_DETECTION_CONFIDENCE,
            tracking_confidence = config.HAND_TRACKING_CONFIDENCE,
        )
        self.classifier = GestureClassifier()

        #Cursor action mapper (only use move_to)
        self.actions = ActionMapper()

        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        #Smoothing memory
        self.prev_screen_xy: Optional[Tuple[int, int]] = None

        #Require a tiny streak for fist to reduce flicker during arming
        self._fist_streak = 0

        #Disarm animation bookkeeping
        self._disarm_active = False
        self._disarm_t0 = 0.0

    def move_pointer(self, landmarks):
        """Map index fingertip (or your chosen pointer) to screen and move cursor."""
        index_xy = self.classifier.get_pointer_position(landmarks)
        if index_xy is None:
            return

        #Map normalized camera coords to screen coords
        screen_xy = self.actions.cursor.map_coordinates(
            index_xy[0],
            index_xy[1],
            config.FRAME_WIDTH,
            config.FRAME_HEIGHT,
        )

        #Smooth movement (tune smoothing in gesture_config if needed)
        screen_xy = smooth(self.prev_screen_xy, screen_xy, alpha=(1 - config.SMOOTHING_FACTOR))
        self.prev_screen_xy = screen_xy

        #Move cursor
        self.actions.ping_action("move_to", screen_xy[0], screen_xy[1], duration=0.0)

    def run(self):
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Cannot read frame from camera")
                    break

                #Mirror the frame (ensure your map_coordinates matches this orientation)
                frame = cv2.flip(frame, 1)

                #Detect hand and draw landmarks
                results = self.detector.detect_hands(frame)
                hand_landmarks = self.detector.get_landmarks(results, frame.shape)
                frame = self.detector.draw_landmarks(frame, results)

                #Default HUD state (for frames without hands)
                hud_state_text = "IDLE"
                hud_progress = 0.0

                if hand_landmarks:
                    landmarks = hand_landmarks[0]

                    #Classify current gesture with classifier
                    gesture = self.classifier.classify_gesture(landmarks)

                    #Normalize for the state machine (expects "fist", "five", or "other")
                    norm = _normalize_for_state(gesture)

                    #Tiny streak to make arming more robust
                    if norm == "fist":
                        self._fist_streak += 1
                    else:
                        self._fist_streak = 0
                    stable_norm = "fist" if self._fist_streak >= 5 else ("five" if norm == "five" else "other")

                    #Hold FIST 3s to arm, when ACTIVE, FIVE moves cursor
                    state = state_machine.update(stable_norm)
                    hud_state_text = state.name

                    #Move cursor only when ACTIVE and gesture is FIVE_FINGERS
                    if state == ControlState.ACTIVE and gesture == GestureClassifier.GESTURE_FIVE_FINGERS:
                        self.move_pointer(landmarks)

                    #Reverse progress bar when disarming (ACTIVE + hold FIST)
                    _hold_secs = getattr(state_machine, "hold_seconds", 3.0)
                    if state == ControlState.ACTIVE:
                        if norm == "fist":
                            # Start disarm timer on first detection of fist
                            if not self._disarm_active:
                                self._disarm_active = True
                                self._disarm_t0 = time.time()

                            #Reverse progress: 1.0 -> 0.0 over _hold_secs
                            elapsed = time.time() - self._disarm_t0
                            hud_progress = max(0.0, 1.0 - (elapsed / _hold_secs))

                            #When progress reaches 0, flip back to IDLE
                            if elapsed >= _hold_secs:
                                if hasattr(state_machine, "force_state"):
                                    state_machine.force_state(ControlState.IDLE)
                                else:
                                    state_machine.current_state = ControlState.IDLE
                                    if hasattr(state_machine, "_timer"):
                                        state_machine._timer = 0.0
                                self._disarm_active = False
                                hud_state_text = "DISARMED"
                                hud_progress = 0.0
                        else:
                            #Any gesture other than fist cancels the disarm
                            self._disarm_active = False

                    #progress bar for arming
                    if hasattr(state_machine, "progress"):
                        try:
                            hud_progress = float(state_machine.progress())  # 0..1 while arming
                        except Exception:
                            hud_progress = 1.0 if state == ControlState.ACTIVE else 0.0
                    else:
                        hud_progress = 1.0 if state == ControlState.ACTIVE else (0.0 if state == ControlState.IDLE else 0.0)

                    #HUD show gesture
                    cv2.putText(
                        frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    )

                #HUD show state and progress
                cv2.putText(
                    frame, f"State: {hud_state_text}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                )
                #Progress bar background adn filters 
                cv2.rectangle(frame, (10, 90), (210, 110), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 90), (10 + int(200 * max(0.0, min(1.0, hud_progress))), 110), (0, 200, 0), -1)
                cv2.imshow("Hand Gesture Cursor (FIST 3s to arm, FIVE to move)", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                time.sleep(0.001)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.detector.cleanup()

if __name__ == "__main__":
    App().run()