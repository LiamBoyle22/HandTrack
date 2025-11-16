from __future__ import annotations
import cv2
import time
from time import monotonic
from typing import Optional, Tuple

from actions.action_mapper import ActionMapper
from gesture_rec.hand_detect import HandDetector
from gesture_rec.gesture_class import GestureClassifier
from gesture_rec import gesture_config as config
from utils.state_machine import GestureStateMachine, ControlState

#Arm after 3s fist, allow brief hiccups during the hold
state_machine = GestureStateMachine(hold_seconds=3.0, drop_grace=0.6)

def smooth(prev: Optional[Tuple[int, int]], curr: Tuple[int, int], alpha: float) -> Tuple[int, int]:
    """EMA smoothing. alpha in [0..1], where 0 keeps 'curr' fully, 1 keeps 'prev' fully."""
    if prev is None:
        return curr
    return (
        int(alpha * prev[0] + (1 - alpha) * curr[0]),
        int(alpha * prev[1] + (1 - alpha) * curr[1]),
    )

def _normalize_for_state(gesture: str) -> str:
    """Map classifier constants to 'fist'/'five'/'other' for the state machine."""
    if gesture == GestureClassifier.GESTURE_FIST:
        return "fist"
    if gesture == GestureClassifier.GESTURE_FIVE_FINGERS:
        return "five"
    return "other"

class App:
    def __init__(self):
        #Hand detector + classifier
        self.detector = HandDetector(
            max_num_hands=config.MAX_NUM_HANDS,
            detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
        )
        self.classifier = GestureClassifier()

        #Cursor/keyboard actions
        self.actions = ActionMapper()

        #Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        #Cursor smoothing
        self.prev_screen_xy: Optional[Tuple[int, int]] = None

        #FIST streak to stabilize arming
        self._fist_streak = 0

        #Disarm animation bookkeeping
        self._disarm_active = False
        self._disarm_t0 = 0.0

        #Debounce one-shot actions
        self._last_fire = {
            "left_click": 0.0,
            "right_click": 0.0,
            "backspace": 0.0,
            "minimize": 0.0,
        }
        self._poke_cooldown_s = 0.45
        self._pinch_cooldown_s = 0.8
        self._pinch_active = False  #rising-edge detector

    def _can_fire(self, key: str, min_interval: float) -> bool:
        now = monotonic()
        if now - self._last_fire.get(key, 0.0) >= min_interval:
            self._last_fire[key] = now
            return True
        return False

    def move_pointer(self, landmarks):
        """Map index fingertip to screen coords and move cursor."""
        index_xy = self.classifier.get_pointer_position(landmarks)
        if index_xy is None:
            return

        #Map normalized camera coords to screen coords using your cursor controller
        screen_xy = self.actions.cursor.map_coordinates(
            index_xy[0],
            index_xy[1],
            config.FRAME_WIDTH,
            config.FRAME_HEIGHT,
        )

        #Smoothing
        screen_xy = smooth(self.prev_screen_xy, screen_xy, alpha=(1 - config.SMOOTHING_FACTOR))
        self.prev_screen_xy = screen_xy

        #Move cursor
        self.actions.ping_action("move_to", screen_xy[0], screen_xy[1], duration=0.0)

    def _handle_pinch_minimize(self, landmarks, state: ControlState):
        """One-shot: pinch => minimize active window (ACTIVE only)."""
        if state != ControlState.ACTIVE:
            self._pinch_active = False
            return

        is_pinch = (self.classifier.classify_gesture(landmarks) == self.classifier.GESTURE_PINCH)
        if is_pinch and not self._pinch_active:
            #rising edge
            if self._can_fire("minimize", self._pinch_cooldown_s):
                self.actions.ping_action("minimize_window")
            self._pinch_active = True
        elif not is_pinch:
            self._pinch_active = False

    def _handle_pokes(self, landmarks, state: ControlState):
        """Depth-based poke gestures (ACTIVE only)."""
        if state != ControlState.ACTIVE:
            return

        #Index poke = left click
        try:
            if self.classifier.detect_poke_index(landmarks):
                if self._can_fire("left_click", self._poke_cooldown_s):
                    self.actions.ping_action("left_click")
        except AttributeError:
            #Poke helpers not present yet, skip silently
            pass

        #Two-finger poke (index+middle) = right click
        try:
            if self.classifier.detect_poke_two_fingers(landmarks):
                if self._can_fire("right_click", self._poke_cooldown_s):
                    self.actions.ping_action("right_click")
        except AttributeError:
            pass

        #Three-finger poke (index+middle+ring) = backspace
        try:
            if self.classifier.detect_poke_three_fingers(landmarks):
                if self._can_fire("backspace", self._poke_cooldown_s):
                    self.actions.ping_action("backspace")
        except AttributeError:
            pass

    def run(self):
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Cannot read frame from camera")
                    break

                #Mirror the frame 
                frame = cv2.flip(frame, 1)

                #Detect hand and draw landmarks (your detector API)
                results = self.detector.detect_hands(frame)
                hand_landmarks = self.detector.get_landmarks(results, frame.shape)
                frame = self.detector.draw_landmarks(frame, results)

                hud_state_text = "IDLE"
                hud_progress = 0.0

                if hand_landmarks:
                    #Use first detected hand
                    landmarks = hand_landmarks[0]

                    #Classify gesture
                    gesture = self.classifier.classify_gesture(landmarks)
                    norm = _normalize_for_state(gesture)

                    #Short streak to make arming more robust
                    if norm == "fist":
                        self._fist_streak += 1
                    else:
                        self._fist_streak = 0
                    stable_norm = "fist" if self._fist_streak >= 5 else ("five" if norm == "five" else "other")

                    #Hold FIST 3s to arm; when ACTIVE, FIVE moves cursor
                    state = state_machine.update(stable_norm)
                    hud_state_text = state.name

                    #Move cursor only when ACTIVE and FIVE_FINGERS is held
                    if state == ControlState.ACTIVE and gesture == GestureClassifier.GESTURE_FIVE_FINGERS:
                        self.move_pointer(landmarks)

                    #One-shot pinch = minimize
                    self._handle_pinch_minimize(landmarks, state)

                    #Poke gestures = click/backspace
                    self._handle_pokes(landmarks, state)

                    #Reverse progress bar when disarming (ACTIVE + hold FIST)
                    _hold_secs = getattr(state_machine, "hold_seconds", 3.0)
                    if state == ControlState.ACTIVE:
                        if norm == "fist":
                            if not self._disarm_active:
                                self._disarm_active = True
                                self._disarm_t0 = time.time()
                            elapsed = time.time() - self._disarm_t0
                            hud_progress = 1.0 - min(1.0, elapsed / _hold_secs)
                            if hud_progress <= 0.0:
                                self._disarm_active = False
                        else:
                            self._disarm_active = False
                    #Arming progress when not ACTIVE
                    if hasattr(state_machine, "progress"):
                        try:
                            hud_progress = float(state_machine.progress())
                        except Exception:
                            hud_progress = 1.0 if state == ControlState.ACTIVE else 0.0
                    else:
                        hud_progress = 1.0 if state == ControlState.ACTIVE else 0.0

                    #HUD show gesture
                    cv2.putText(
                        frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    )

                else:
                    #No hand: keep state machine ticking with 'other'
                    state = state_machine.update("other")
                    hud_state_text = state.name
                    hud_progress = getattr(state_machine, "arming_progress", 0.0) if hasattr(state_machine, "arming_progress") else 0.0
                    self._fist_streak = 0
                    self._pinch_active = False

                #HUD state + progress bar
                cv2.putText(frame, f"State: {hud_state_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,)

                #Progress bar
                cv2.rectangle(frame, (10, 90), (210, 110), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 90), (10 + int(200 * max(0.0, min(1.0, hud_progress))), 110), (0, 200, 0), -1,)

                cv2.imshow("Hand Gesture Cursor (FIST 3s to arm, FIVE to move)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                time.sleep(0.001)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            try:
                self.detector.cleanup()
            except Exception:
                pass

if __name__ == "__main__":
    App().run()
