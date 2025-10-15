import cv2
import mediapipe as mp
import pyautogui 
from gesture_controller import GestureController, execute_action

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.drawing_styles
mphands = mp.solutions.hands

def main():
    cap = cv2.VideoCapture(0)
    hands = mphands.Hands(
        max_num_hands = 1,
        min_detection_confedince = 0.7,
        min_tracking_confedince = 0.7,
    )

    controller = GestureController()

    pyautogui.FAILSAFE = True

    print("Hand gesture control satarted")
    print("Press 'q' to quit")
    print("\nGesture Controls:")
    print("- POINT (index finger up): Move cursor")
    print("- Pinch (thumb and index finger together): Left click")
    print("- SCROLL (peace sign): Scroll up/down")
    print("- GRAB (fist): select text")
    print("- L-SHAPE (L shape with thumb and index finger): Undo")
    print("- PALM (open hand): Copy")
    print("- THUMB UP (thumb up): Paste")

    while True:
        data, image = cap.read()
        if not data:
            print("failed to capture video")
            break

        frame_height, frame_width = image.shape
        image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        action_text = "No hand detected"

        if results.multi_hand_landmarks:
            for hands_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hands_landmarks,
                    mphands.HAND_CONNECTIONS
                )

                gesture = controller.detect-gesture(hands_landmarks.landmark)

                action_result = execute_action(
                    gesture,
                    controller,
                    hands_landmarks.landmark,
                    frame_width,
                    frame_height
                )

                action_text = f"Gesture: {gesture}, Action: {action_result}"

        cv2.putText(image, action_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hand Gesture Control", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()