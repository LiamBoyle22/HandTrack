import math
import numpy as np
import gesture_config as config
from typing import Sequence

class GestureClassifier:

    GESTURE_NONE = "None"
    GESTURE_POINTER = "Pointer"
    GESTURE_PINCH = "Pinch"
    GESTURE_PEACE = "Peace"
    GESTURE_THREE_FINGERS = "ThreeFingers"
    GESTURE_FOUR_FINGERS = "FourFingers"
    GESTURE_FIVE_FINGERS = "FiveFingers"
    GESTURE_FIST = "Fist"
    GESTURE_THUMBS_UP = "ThumbsUp"

    def __init__ (self):
        self.current_gesture = self.GESTURE_NONE
        self.previous_gesture = self.GESTURE_NONE
        self.gesture_start_frame = 0
        self.frame_count = 0

        self.cooldown_frames = config.GESTURE_COOLDOWN_FRAMES
        self.last_gesture_frame = self.cooldown_frames

    POKE_Z_DELTA = getattr(config, "POKE_Z_DELTA", 0.08) #depth from wrist to finger tip to count as poke
    POKE_REQUIRE_EXTENDED = getattr(config, "POKE_REQUIRE_EXTENDED", True)

    def z(self, pt):
        #Fall back to 0.0 if missing
        return pt.get("z", 0.0)

    def _tips_for(self, landmarks, finger_names: Sequence[str]):
        m = {
            "thumb":  config.HandLandmark.THUMB_TIP,
            "index":  config.HandLandmark.INDEX_FINGER_TIP,
            "middle": config.HandLandmark.MIDDLE_FINGER_TIP,
            "ring":   config.HandLandmark.RING_FINGER_TIP,
            "pinky":  config.HandLandmark.PINKY_TIP,
        }
        return [landmarks[m[name]] for name in finger_names]
    
    def is_finger_extended(self, landmarks, finger_tip_idx, finger_pip_idx):
        tip = landmarks[finger_tip_idx]
        pip = landmarks[finger_pip_idx]
        wrist = landmarks[config.HandLandmark.WRIST]

        tip_to_wrist = self.calc_distance(tip, wrist)
        pip_to_wrist = self.calc_distance(pip, wrist)

        return tip_to_wrist > (pip_to_wrist + 0.02)
    
    def is_thumb_extended(self, landmarks):
        thumb_tip = landmarks[config.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[config.HandLandmark.THUMB_IP]
        index_mcp = landmarks[config.HandLandmark.INDEX_FINGER_MCP]

        tip_distance = self.calc_distance(thumb_tip, index_mcp)
        ip_distance = self.calc_distance(thumb_ip, index_mcp)

        return tip_distance > ip_distance
    
    def is_pinch(self, landmarks):
        thumb_tip = landmarks[config.HandLandmark.THUMB_TIP]
        index_tip = landmarks[config.HandLandmark.INDEX_FINGER_TIP]
        
        distance = self.calc_distance(thumb_tip, index_tip)

        wrist = landmarks[config.HandLandmark.WRIST]
        middle_mcp = landmarks[config.HandLandmark.MIDDLE_FINGER_MCP]
        hand_size = self.calc_distance(wrist, middle_mcp)

        normalized_distance = distance / hand_size if hand_size > 0 else distance
        
        return normalized_distance < config.PINCH_THRESHOLD
    
    def is_fist(self, landmarks):
        finger_info = self.extended_fingers(landmarks)
        
        return finger_info['count'] == 0 or (finger_info['count'] == 1 and finger_info['thumb'])
    
    def extended_fingers(self, landmarks):
        thumb_extended = self.is_thumb_extended(landmarks)
        index_extended = self.is_finger_extended(landmarks, config.HandLandmark.INDEX_FINGER_TIP, config.HandLandmark.INDEX_FINGER_PIP)
        middle_extended = self.is_finger_extended(landmarks, config.HandLandmark.MIDDLE_FINGER_TIP, config.HandLandmark.MIDDLE_FINGER_PIP)
        ring_extended = self.is_finger_extended(landmarks, config.HandLandmark.RING_FINGER_TIP, config.HandLandmark.RING_FINGER_PIP)
        pinky_extended = self.is_finger_extended(landmarks, config.HandLandmark.PINKY_TIP, config.HandLandmark.PINKY_PIP)
        
        count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
        
        return {
            'count': count,
            'thumb': thumb_extended,
            'index': index_extended,
            'middle': middle_extended,
            'ring': ring_extended,
            'pinky': pinky_extended
        }

    def _all_extended(self, landmarks, finger_names: Sequence[str]) -> bool:
        fi = self.extended_fingers(landmarks)
        want = {
            "thumb": fi["thumb"],
            "index": fi["index"],
            "middle": fi["middle"],
            "ring": fi["ring"],
            "pinky": fi["pinky"],
        }
        return all(want[name] for name in finger_names)

    def _is_poke(self, landmarks, finger_names: Sequence[str]) -> bool:

        if not landmarks or len(landmarks) != 21:
            return False

        if self.POKE_REQUIRE_EXTENDED and not self._all_extended(landmarks, finger_names):
            return False

        wrist_z = self.z(landmarks[config.HandLandmark.WRIST])
        tips = self._tips_for(landmarks, finger_names)

        #Trigger when all selected tips are closer than wrist by a delta.
        return all((self.z(t) - wrist_z) < (-self.POKE_Z_DELTA) for t in tips)
    def poke_index(self, landmarks) -> bool:
        return self._is_poke(landmarks, ["index"])

    def poke_two_fingers(self, landmarks) -> bool:
        return self._is_poke(landmarks, ["index", "middle"])

    def poke_three_fingers(self, landmarks) -> bool:
        return self._is_poke(landmarks, ["index", "middle", "ring"])

    def calc_distance(self, point1, point2): #short for calculate, im just using slang
        dx = point1['x'] - point2['x']
        dy = point1['y'] - point2['y']
        return math.sqrt(dx * dx + dy * dy)
    
    def classify_gesture(self, landmarks):
        if not landmarks or len(landmarks) != 21:
            return self.GESTURE_NONE
        
        finger_info = self.extended_fingers(landmarks)

        if self.is_pinch(landmarks):
            return self.GESTURE_PINCH
        
        if self.is_fist(landmarks):
            return self.GESTURE_FIST
        
        if finger_info['count'] == 1:
            if finger_info['index'] and not finger_info['thumb']:
                return self.GESTURE_POINTER
            elif finger_info['thumb'] and not finger_info['index']:
                return self.GESTURE_THUMBS_UP
            
        elif finger_info['count'] == 2:
            if finger_info['index'] and finger_info['middle']:
                return self.GESTURE_PEACE
            
        elif finger_info['count'] == 3:
            if finger_info['index'] and finger_info['middle'] and finger_info['ring']:
                return self.GESTURE_THREE_FINGERS
            
        elif finger_info['count'] == 4:
            if not finger_info['thumb']:
                return self.GESTURE_FOUR_FINGERS
            
        elif finger_info['count'] == 5:
            return self.GESTURE_FIVE_FINGERS
    
        return self.GESTURE_NONE
    def update_gesture(self, landmarks):
        self.frame_count += 1
        gesture = self.classify_gesture(landmarks)
        frames_since_last = self.frame_count - self.last_gesture_frame
        is_new_gesture = False

        if gesture != self.GESTURE_NONE:
            if gesture != self.current_gesture or frames_since_last >= self.cooldown_frames:
                is_new_gesture = True
                self.previous_gesture = self.current_gesture
                self.current_gesture = gesture
                self.gesture_start_frame = self.frame_count
                self.last_gesture_frame = self.frame_count

            else:
                if self.current_gesture != self.GESTURE_NONE:
                    self.previous_gesture = self.current_gesture
                self.current_gesture = self.GESTURE_NONE
            
            return (self.current_gesture, is_new_gesture)
        
    def pointer_position(self, landmarks):

        if not landmarks:
            return None

        index_tip = landmarks[config.HandLandmark.INDEX_FINGER_TIP]
        return (index_tip['x'], index_tip['y'])
    
    def gesture_info(self):
        return {
            'current_gesture': self.current_gesture,
            'previous_gesture': self.previous_gesture,
            'frame_since_last_gesture': self.frame_count - self.last_gesture_frame,
            'frame_count': self.frame_count
        }