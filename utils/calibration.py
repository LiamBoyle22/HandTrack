#calibration for visualization
import numpy as np
import json
import os

class HandCalibration:

    def __init__(self):
        
        #hand size
        self.hand_length = None
        self.palm_width = None

        #movement boundaries
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

        #gesture thresholds
        self.pinch_threshold = None
        self.finger_extension_threshold = None

        #calibration state
        self.is_calibrated = False
        self.sample_count = 0
        self.max_samples = 100

        #buffers for calibration data
        self.x_positions = []
        self.y_positions = []
        self.hand_sizes = []

    def calc_hand_size(self, landmarks):
        if not landmarks or len(landmarks) < 21:
            return None
        
        wrist = np.array(landmarks[0])
        middle_tip = np.array(landmarks[12])
        index_mcp = np.array(landmarks[5])
        pinky_mcp = np.array(landmarks[17])

        hand_length = np.sqrt((middle_tip['x'] - wrist['x'])**2 + (middle_tip['y'] - wrist['y'])**2)
        palm_width = np.sqrt((pinky_mcp['x'] - index_mcp['x'])**2 + (pinky_mcp['y'] - index_mcp['y'])**2)
        
        return {
            'hand_length': hand_length,
            'palm_width': palm_width
        }
    
    def add_calibration_sample(self, landmarks):
        if not landmarks or len(landmarks) != 21:
            return False 
        
        hand_size = self.calc_hand_size(landmarks)
        if hand_size:
            self.hand_sizes.append(hand_size['hand_length'])

        index_tip = landmarks[8]
        self.x_positions.append(index_tip['x'])
        self.y_positions.append(index_tip['y'])
        self.sample_count += 1

        if self.sample_count >= self.max_samples:
            self.finalize_calibration()
            return True
        
        return False
    
    def finalize_calibration(self):
        if not self.hand_sizes or not self.x_positions:
            return
        
        self.hand_length = np.mean(self.hand_sizes)

        #movement range with padding
        x_padding = (max(self.x_positions) - min(self.x_positions)) * 0.1
        y_padding = (max(self.y_positions) - min(self.y_positions)) * 0.1

        self.min_x = min(self.x_positions) - x_padding
        self.max_x = max(self.x_positions) + x_padding
        self.min_y = min(self.y_positions) - y_padding
        self.max_y = max(self.y_positions) + y_padding

        #pinch threshold as 10% of hand length
        self.pinch_threshold = self.hand_length * 0.1

        #finger extension threshold as 50% of hand length
        self.finger_extension_threshold = self.hand_length * 0.5

        self.is_calibrated = True

    def map_to_screen(self, x, y, screen_width, screen_height):
        
        if not self.is_calibrated:
            return (int(x * screen_width), int(y * screen_height))
        
        x = map(self.min-x, min(self.max_x, x))
        y = map(self.min_y, max(self.max_y, y))

        x_norm = (x - self.min_x) / (self.max_x - self.min_x)
        y_norm = (y - self.min_y) / (self.max_y - self.min_y)

        screen_x = int(x_norm * screen_width)
        screen_y = int(y_norm * screen_height)

        screen_x = max(0, min(screen_width - 1, screen_x))
        screen_y = max(0, min(screen_height - 1, screen_y))

        return (screen_x, screen_y)
    
    def get_scaled_threshold(self, base_threshold):
        
        if not self.is_calibrated or self.hand_length is None:
            return base_threshold
        
        average_hand_length = 200.0
        scale_factor = self.hand_length / average_hand_length

        return base_threshold * scale_factor
    
    def get_progress(self):

        return min(1.0, self.sample_count / self.max_samples)
    
    def reset(self):
        self.hand_length = None
        self.palm_width = None
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.pinch_threshold = None
        self.finger_extension_threshold = None
        self.is_calibrated = False
        self.sample_count = 0
        self.x_positions = []
        self.y_positions = []
        self.hand_sizes = []

    def save(self, filename = 'calibration.json'):
        if not self.is_calibrated:
            return False
        
        data = {
            'hand_length': self.hand_length,
            'palm_width': self.palm_width,
            'min_x': self.min_x,
            'max_x': self.max_x,
            'min_y': self.min_y,
            'max_y': self.max_y,
            'pinch_threshold': self.pinch_threshold,
            'finger_extension_threshold': self.finger_extension_threshold
        }

        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            return False
        
    def load(self, filename = 'calibration.json'):
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.hand_length = data['hand_length']
            self.palm_width = data['palm_width']
            self.min_x = data['min_x']
            self.max_x = data['max_x']
            self.min_y = data['min_y']
            self.max_y = data['max_y']
            self.pinch_threshold = data['pinch_threshold']
            self.finger_extension_threshold = data['finger_extension_threshold']
            self.is_calibrated = True
            self.sample_count = self.max_samples

            return True
        
        except Exception as e:
            return False
        
class QuickCalibration:

    def __init__(self, num_smaples = 10):
        self.num_smaples = num_smaples
        self.samples = []
        self.hand_length = None
        self.is_complete = False

    def add_sample(self, landmarks):
        if not landmarks or len(landmarks) != 21:
            return False
        
        wrist = landmarks[0]
        middle_tip = landmarks[12]

        length = np.sqrt((middle_tip['x'] - wrist['x'])**2 + (middle_tip['y'] - wrist['y'])**2)
        self.samples.append(length)

        if len(self.smaples) >= self.num_smaples:
            self.hand_length = np.mean(self.samples)
            self.is_complete = True
            return True
        
        return False
    
    def get_progress(self):
        return min(1.0, len(self.samples) / self.num_smaples)