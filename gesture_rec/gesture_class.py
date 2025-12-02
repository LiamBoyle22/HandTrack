import math
import numpy as np
from gesture_rec import gesture_config as config
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
    GESTURE_THUMBS_DOWN = "ThumbsDown"

    def __init__ (self):
        self.current_gesture = self.GESTURE_NONE
        self.previous_gesture = self.GESTURE_NONE
        self.gesture_start_frame = 0
        self.frame_count = 0

        self.cooldown_frames = config.GESTURE_COOLDOWN_FRAMES
        self.last_gesture_frame = self.cooldown_frames

    POKE_Z_DELTA = getattr(config, "POKE_Z_DELTA", 0.08) #depth from wrist to finger tip to count as poke
    POKE_REQUIRE_EXTENDED = getattr(config, "POKE_REQUIRE_EXTENDED", True)
    THUMBS_Y_DELTA = getattr(config, "THUMBS_Y_DELTA", 0.10)

    def z(self, pt):
        #Fall back to 0.0 if missing
        return pt.get("z", 0.0)

    def tips_for(self, landmarks, finger_names: Sequence[str]):
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
    
    def _xy(self, lm):
        #MediaPipe style object
        if hasattr(lm, "x") and hasattr(lm, "y"):
            return lm.x, lm.y

        #Dict style
        if isinstance(lm, dict):
            if "x" in lm and "y" in lm:
                return lm["x"], lm["y"]
            if 0 in lm and 1 in lm:
                return lm[0], lm[1]
            raise KeyError("Landmark dict missing 'x'/'y' or numeric 0/1 keys")

        #Sequence style (tuple/list)
        return lm[0], lm[1]
    
    def _dist(self, landmarks, i, j):
        xi, yi = self._xy(landmarks[i])
        xj, yj = self._xy(landmarks[j])
        return math.hypot(xi - xj, yi - yj)

    def _is_thumb_extended_geom(self, landmarks):
        WRIST = 0
        THUMB_MCP = 2
        THUMB_TIP = 4

        d_tip = self._dist(landmarks, WRIST, THUMB_TIP)
        d_mcp = self._dist(landmarks, WRIST, THUMB_MCP)

        return d_tip > d_mcp * 1.1

    def _is_finger_extended_geom(self, landmarks, tip_idx, mcp_idx):
        WRIST = 0
        d_tip = self._dist(landmarks, WRIST, tip_idx)
        d_mcp = self._dist(landmarks, WRIST, mcp_idx)

        return d_tip > d_mcp * 1.1

    def thumb_direction(self, landmarks):
        if not landmarks or len(landmarks) < 6:
            return "none"

        #Mediapipe indices
        WRIST = 0
        THUMB_MCP = 2  
        THUMB_TIP = 4
        INDEX_MCP = 5

        wx, wy = self._xy(landmarks[WRIST])
        tx, ty = self._xy(landmarks[THUMB_TIP])
        mx, my = self._xy(landmarks[THUMB_MCP])
        ix, iy = self._xy(landmarks[INDEX_MCP])

        #Rough hand size to make threshold scale with distance to camera
        hand_size = ( (ix - wx) ** 2 + (iy - wy) ** 2 ) ** 0.5
        if hand_size < 1e-6:
            hand_size = 1.0

        dy = ty - my  
        threshold = 0.25 * hand_size  

        if dy < -threshold:
            return "up"
        elif dy > threshold:
            return "down"
        else:
            return "none"

        
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
        return finger_info['count'] == 0

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

    def all_extended(self, landmarks, finger_names: Sequence[str]) -> bool:
        fi = self.extended_fingers(landmarks)
        want = {
            "thumb": fi["thumb"],
            "index": fi["index"],
            "middle": fi["middle"],
            "ring": fi["ring"],
            "pinky": fi["pinky"],
        }
        return all(want[name] for name in finger_names)

    def is_poke(self, landmarks, finger_names: Sequence[str]) -> bool:

        if not landmarks or len(landmarks) != 21:
            return False

        if self.POKE_REQUIRE_EXTENDED and not self.all_extended(landmarks, finger_names):
            return False

        wrist_z = self.z(landmarks[config.HandLandmark.WRIST])
        tips = self.tips_for(landmarks, finger_names)

        #Trigger when all selected tips are closer than wrist by a delta
        return all((self.z(t) - wrist_z) < (-self.POKE_Z_DELTA) for t in tips)
    def poke_index(self, landmarks) -> bool:
        return self.is_poke(landmarks, ["index"])

    def poke_two_fingers(self, landmarks) -> bool:
        return self.is_poke(landmarks, ["index", "middle"])

    def poke_three_fingers(self, landmarks) -> bool:
        return self.is_poke(landmarks, ["index", "middle", "ring"])

    def calc_distance(self, point1, point2): #short for calculate, im just using slang
        dx = point1['x'] - point2['x']
        dy = point1['y'] - point2['y']
        return math.sqrt(dx * dx + dy * dy)
    
    def classify_gesture(self, landmarks):
        if not landmarks or len(landmarks) != 21:
            return self.GESTURE_NONE

        finger_info = self.extended_fingers(landmarks)

        thumb_ext = self._is_thumb_extended_geom(landmarks)
        index_ext = self._is_finger_extended_geom(landmarks, tip_idx=8,  mcp_idx=5)
        middle_ext = self._is_finger_extended_geom(landmarks, tip_idx=12, mcp_idx=9)
        ring_ext = self._is_finger_extended_geom(landmarks, tip_idx=16, mcp_idx=13)
        pinky_ext = self._is_finger_extended_geom(landmarks, tip_idx=20, mcp_idx=17)

        thumb_dir = self.thumb_direction(landmarks)

        if hasattr(self, "is_pinch") and self.is_pinch(landmarks):
            return self.GESTURE_PINCH

        if hasattr(self, "is_fist") and self.is_fist(landmarks):
            return self.GESTURE_FIST

        if (
            thumb_ext
            and not index_ext
            and not middle_ext
            and not ring_ext
            and not pinky_ext
        ):
            if thumb_dir == "up":
                return self.GESTURE_THUMBS_UP
            elif thumb_dir == "down":
                return self.GESTURE_THUMBS_DOWN
            else:
                return self.GESTURE_NONE

        if finger_info.get("count", 0) == 5:
            return self.GESTURE_FIVE_FINGERS

        if finger_info.get("count", 0) == 1:
            if finger_info.get("index") and not finger_info.get("thumb"):
                return self.GESTURE_POINTER

        if finger_info.get("count", 0) == 2:
            if finger_info.get("index") and finger_info.get("middle"):
                return self.GESTURE_PEACE

        if finger_info.get("count", 0) == 3:
            if finger_info.get("index") and finger_info.get("middle") and finger_info.get("ring"):
                return self.GESTURE_THREE_FINGERS

        if finger_info.get("count", 0) == 4:
            if not finger_info.get("thumb"):
                return self.GESTURE_FOUR_FINGERS

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