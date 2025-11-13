from __future__ import annotations
import cv2
import time
from typing import Optional, Tuple

from actions.action_mapper import ActionMapper
from gesture_rec.hand_detect import HandDetector
from gesture_rec.gesture_class import GestureClassifier
from gesture_rec import gesture_config as config

def smooth(prev: Optional[Tuple[int, int]], curr: Tuple[int, int], alpha: float = 0.7) -> Tuple[int, int]:
    if prev is None:
        return curr
    return (
        int(alpha * prev[0] + (1 - alpha) * curr[0]),
        int(alpha * prev[1] + (1 - alpha) * curr[1]),
    )

class App:
    def __init__(self):
        self.detector = HandDetector(
            max_num_hands = config.MAX_NUM_HANDS,
            detection_confidence = config.HAND_DETECTION_CONFIDENCE,
            tracking_confidence = config.HAND_TRACKING_CONFIDENCE,
        )
        self.classifier = GestureClassifier()
        self.actions = ActionMapper()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        self.prev_screen_xy: Optional[Tuple[int, int]] = None
        self.drag_active = False

    def move_pointer(self, landmarks):
        index_xy = self.classifier.get_pointer_position(landmarks)
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

        self.actions.ping_action("move_to", screen_xy[0], screen_xy[1], duration = 0.0)

    def handle_gesture_actions(self, gesture: str):
        if gesture == GestureClassifier.GESTURE_PINCH:
            if not self.drag_active:
                self.actions.ping_action("mouse_down", button = "left")
                self.drag_active = True

        else:
            if self.drag_active:
                self.actions.ping_action("mouse_up", button = "left")
                self.drag_active = False

        if gesture == GestureClassifier.GESTURE_FIST:
            pass #FOR NOW BECAUSE MOVE POINTER HANDLES IT 
        elif gesture == GestureClassifier.GESTURE_PEACE:
            self.actions.ping_action("left_click")
        elif gesture == GestureClassifier.GESTURE_THREE_FINGERS:
            self.actions.ping_action("right_click")
        elif gesture == GestureClassifier.GESTURE_FOUR_FINGERS:
            self.actions.ping_action("scroll_down", config.SCROLL_AMOUNT)
        elif gesture == GestureClassifier.GESTURE_FIVE_FINGERS:
            self.actions.ping_action("scroll_up", config.SCROLL_AMOUNT)
        elif gesture == GestureClassifier.GESTURE_THUMBS_UP:
            self.actions.ping_action("select_all")

    def run (self):
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Cannot read frame from camera")
                    break
                    
                frame = cv2.flip(frame, 1)

                results = self.detector.detect_hands(frame)
                hand_landmarks = self.detector.get_landmarks(results, frame.shape)

                frame = self.detector.draw_landmarks(frame, results)

                if hand_landmarks:
                    landmarks = hand_landmarks[0]

                    gesture = self.classifier.classify_gesture(landmarks)

                    if gesture == GestureClassifier.GESTURE_FIST:
                        self.move_pointer(landmarks)

                    self.handle_gesture_actions(gesture)


                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2,
                    )

                cv2.imshow("Hand Gesture Control", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break 

                time.sleep(0.001)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.detector.cleanup()

if __name__ == "__main__":
    App().run() 