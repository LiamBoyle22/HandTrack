from __future__ import annotations
import cv2
import time
from typing import Optional, Tuple
import math

from actions.action_mapper import ActionMapper
from gesture_rec.hand_detect import HandDetector
from gesture_rec.gesture_class import GestureClassifier
from gesture_rec import gesture_config as config
from utils.state_machine import GestureStateMachine, ControlState

#testing 
SCROLL_STEP = 120 #Typical scroll step value

def smooth(prev: Optional[Tuple[int, int]], curr: Tuple[int, int], alpha: float) -> Tuple[int, int]:
    if prev is None:
        return curr
    return (
        int(alpha * prev[0] + (1 - alpha) * curr[0]),
        int(alpha * prev[1] + (1 - alpha) * curr[1]),
    )


#One Euro filter for later 
class OneEuroFilter:
    def __init__(self, freq=60.0, min_cutoff=1.5, beta=0.35, d_cutoff=20.0):
        self.freq = freq  # expected samples/sec
        self.min_cutoff = min_cutoff
        self.beta = beta  # velocity sensitivity
        self.d_cutoff = d_cutoff
        self.last_time = None
        self.prev_x = None
        self.prev_dx = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / max(1e-3, self.freq)
        return 1.0 / (1.0 + tau / te)

    def OneEuroF(self, x):
        #Estimate dx
        if self.prev_x is None:
            dx = 0.0
        else:
            dx = (x - self.prev_x) * self.freq

        #Smooth dx
        a_d = self.alpha(self.d_cutoff)
        dx_hat = dx if self.prev_dx is None else (a_d * dx + (1 - a_d) * self.prev_dx)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = x if self.prev_x is None else (a * x + (1 - a) * self.prev_x)
        self.prev_x = x_hat
        self.prev_dx = dx_hat
        return x_hat


class HTApp:
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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        #Cursor smoothing
        self.prev_screen_xy: Optional[Tuple[int, int]] = None

        #Gesture state machine (ThumbsUp / ThumbsDown -> ACTIVE / IDLE)
        self.state_machine = GestureStateMachine()

        self.finger_hold_start_time = None
        self.finger_hold_click_fired = False
        self.FINGER_HOLD_TIME = 1.0  # seconds


    def move_pointer(self, landmarks):
        index_xy = self.classifier.pointer_position(landmarks)
        if index_xy is None:
            return

        screen_xy = self.actions.cursor.map_coordinates(
            index_xy[0],
            index_xy[1],
            config.FRAME_WIDTH,
            config.FRAME_HEIGHT,
        )
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

                #Mirror img
                frame = cv2.flip(frame, 1)

                #Detect hand and landmarks
                results = self.detector.detect_hands(frame)
                hand_landmarks = self.detector.get_landmarks(results, frame.shape)
                frame = self.detector.draw_landmarks(frame, results)

                #HUD based on current state
                state = self.state_machine.state
                hud_state_text = "ACTIVE" if state == ControlState.ACTIVE else "IDLE"
                hud_progress = self.state_machine.progress()

                if hand_landmarks:
                    #Use first detected hand
                    landmarks = hand_landmarks[0]

                    #Classify gesture
                    gesture = self.classifier.classify_gesture(landmarks)

                    #Update state machine with this gesture
                    state = self.state_machine.update(gesture)

                    #Cursor moves only when ACTIVE and hand is five fingers
                                        # Cursor moves only when ACTIVE and hand is five fingers
                    if (
                        state == ControlState.ACTIVE
                        and gesture == GestureClassifier.GESTURE_FIVE_FINGERS
                    ):
                        self.move_pointer(landmarks)

                    if state == ControlState.ACTIVE:
                        if gesture == GestureClassifier.GESTURE_THREE_FINGERS_UP:
                            self.actions.ping_action("scroll_up", SCROLL_STEP)
                        elif gesture == GestureClassifier.GESTURE_THREE_FINGERS_DOWN:
                            self.actions.ping_action("scroll_down", SCROLL_STEP)

                    if state == ControlState.ACTIVE and gesture == GestureClassifier.GESTURE_POINTER:
                        now = time.time()
                        if self.finger_hold_start_time is None:
                            self.finger_hold_start_time = now
                            self.finger_hold_click_fired = False
                        else:
                            if (
                                not self.finger_hold_click_fired
                                and (now - self.finger_hold_start_time) >= self.FINGER_HOLD_TIME
                            ):
                                self.actions.ping_action("left_click")
                                self.finger_hold_click_fired = True
                    else:
                        self.finger_hold_start_time = None
                        self.finger_hold_click_fired = False

                    #HUD: show current gesture text
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2,
                    )

                    #Refresh HUD after state update
                    hud_state_text = "ACTIVE" if state == ControlState.ACTIVE else "IDLE"
                    hud_progress = self.state_machine.progress()

                #HUD state text
                cv2.putText(
                    frame,
                    f"State: {hud_state_text}",
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                )

                #Progress bar (simple ON/OFF, but kept for compatibility)
                cv2.rectangle(frame, (10, 90), (210, 110), (40, 40, 40), -1)
                cv2.rectangle(
                    frame,
                    (10, 90),
                    (10 + int(200 * max(0.0, min(1.0, hud_progress))), 110),
                    (0, 200, 0),
                    -1,
                )

                cv2.imshow("Hand Gesture Cursor (Thumbs Up=ON, Thumbs Down=OFF, FIVE to move)", frame)
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
    HTApp().run()