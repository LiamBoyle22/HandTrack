import numpy as np 
import pyautogui 
import time
from collections import deque

class GestureController:
    def __init__(self):
        self.prev_gesture = None
        self.gesture_start_time = None
        self.gesture_threshold = 0.5
        self.smoothing_buffer = deque(maxlen=5)
        self.scroll_sensitivity = 5

    def get_finger_states(self, landmarks):
        fingers = []

        if landmarks[4].x < landmarks[3].x:
            fingers.append(True)
        else:
            fingers.append(False)

        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                fingers.append(True)
            else:
                fingers.append(False)
        return fingers
    
    def calculate_distance(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def detect_gesture(self, landmarks):
        fingers = self.get_finger_states(landmarks)
    
        if fingers == [False, True, False, False, False]:
            return "POINT"
        
        if fingers == [True, True, False, False, False]:
            return "SCROLL"
        
        if fingers == [False, False, False, False, False]:
            return "GRAB"
        
        if fingers == [False, True, True, False, False]:
            return "PEACE"
        
        if fingers == [True, False, False, False, False]:
            return "THUMB_UP"
        
        if fingers == [True, True, False, False, True]:
            return "L_SHAPE"
        
        if all(fingers):
            return "PALM"
        
        return "UNKNOWN"
    
    def get_cursor_poition(self, landmarks, frame_width, frame_height):
        index_finger_tip = landmarks[8]
        x = int(index_finger_tip.x * frame_width)
        y = int(index_finger_tip.y * frame_height)
        
        screen_w, screen_h = pyautogui.size()
        screen_x = np.interp(x, (0, frame_width), (0, screen_w))
        screen_y = np.interp(y, (0, frame_height), (0, screen_h))

        self.smoothing_buffer.append((screen_x, screen_y))
        avg_x = sum(p[0] for p in self.smoothing_buffer) / len(self.smoothing_buffer)
        avg_y = sum(p[1] for p in self.smoothing_buffer) / len(self.smoothing_buffer)
        return int(avg_x), int(avg_y)
    
    def execute_action(self, gesture, landmarks, frame_width, frame_height):