import cv2
import mediapipe
import numpy as np
import gesture_config as config

class HandDetector:
    def __init__(self,
                 max_num_hands = config.MAX_NUM_HANDS,
                 detection_confidence = config.HAND_DETECTION_CONFIDENCE,
                 tracking_confidence = config.HAND_TRACKING_CONFIDENCE):

        self.mp_hands = mediapipe.solutions.hands
        self.mp_drawing = mediapipe.solutions.drawing_utils
        self.mp_drawing_styles = mediapipe.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                          max_num_hands=max_num_hands,
                                          min_detection_confidence=detection_confidence,
                                          min_tracking_confidence=tracking_confidence)
        
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

    def detect_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results
    
    def get_landmarks(self, results, frame_shape):
        if not results.multi_hand_landmarks:
            return []
        
        height, width, _ = frame_shape
        hands_landmarks = []

        for i in range(len(results.multi_hand_landmarks)):
            hand_landmarks = results.multi_hand_landmarks[i]
            landmarks = []  
            for landmark in hand_landmarks.landmark:
                landmark_dict = {
                    'x': int(landmark.x * width),
                    'y': int(landmark.y * height),
                    'z': landmark.z,
                    'relative_x': landmark.x,
                    'relative_y': landmark.y,
                    'visibility': landmark.visibility
                }
                landmarks.append(landmark_dict)
            hands_landmarks.append(landmarks)
        return hands_landmarks
    
    def get_hand_info(self, results):
        if not results.multi_handedness:
            return []
        hand_info = []
        for i in range(len(results.multi_handedness)):
            hand = results.multi_handedness[i]
            handedness = hand.classification[0].label  # Use 'hand', not 'hand_info'
            score = hand.classification[0].score        # Use 'hand', not 'hand_info'

            hand_info.append({
                'handedness': handedness,
                'score': score
            })
        return hand_info
    
    def draw_landmarks(self, frame, results,
                       draw_landmarks = config.SHOW_LANDMARKS,
                          draw_connections = config.SHOW_CONNECTIONS):
        
        if not results.multi_hand_landmarks:
            return frame
        
        for i in range(len(results.multi_hand_landmarks)):
            hand_landmarks = results.multi_hand_landmarks[i]
            if draw_landmarks and draw_connections:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
            
            elif draw_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    None,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    )
                
        return frame 
    
    def draw_bounding_box(self, frame, landmarks):
        if not landmarks:
            return frame
        
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return frame
    
    def get_finger_tip_position(self, landmarks, finger_tip_index):
        if not landmarks or finger_tip_index >= len(landmarks):
            return None

        landmark = landmarks[finger_tip_index]
        return (int(landmark['x']), int(landmark['y']))
    
    def cleanup(self):
        self.hands.close()

if __name__ == "__main__":
    detector = HandDetector()
    cap = cv2.VideoCapture(0)


    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        results = detector.detect_hands(frame)
        hands_landmarks = detector.get_landmarks(results, frame.shape)
        hands_info = detector.get_hand_info(results)

        frame = detector.draw_landmarks(frame, results)

        if hands_landmarks:
            cv2.putText(frame, f'Hands Detected: {len(hands_landmarks)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for i, info in enumerate(hands_info):
            cv2.putText(frame, f'Hand {i+1}: {info["handedness"]} ({info["score"]:.2f})', (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()
