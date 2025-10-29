import cv2
import time
import numpy as np

#To-Do
    #Import modules from other files

class GestctrlSys:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            raise Exception("Could not open video device, is it connected?")
        
        #1920 by 1080 for cursor mapping
        self.screen_width = 1920
        self.screen_height = 1080

        #FPS tracking
        self.prev_time = 0
        self.curr_time = 0

        #shows FPS and debug info
        self.is_running = True
        self.show_debug = True
        
        #To-Do: initialize other modules here
        
    def calculate_fps(self):
        self.curr_time = time.time()
        fps = 1/(self.curr_time - self.prev_time)
        self.prev_time = self.curr_time
        return (fps)
    
    def process_frame(self, frame):
        #To-Do: Add gesture recognition and cursor control logic here
        return frame
    
    def draw_debug_info(self, frame, fps):
        if self.show_debug:
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, "Press q to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def run(self):
        try:
            while self.is_running:
                sucess, frame = self.cap.read()

                if not sucess:
                    print("Failed to grab frame")
                    break

                frame = cv2.flip(frame, 1)
                fps = self.calculate_fps()
                pframe = self.process_frame(frame)
                self.draw_debug_info(pframe, fps)
                cv2.imshow("Gesture Control System", pframe)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.is_running = False

                elif key == ord('d'):
                    self.show_debug = not self.show_debug

        except KeyboardInterrupt:
            print("Interrupted by user")

        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    
    try: 
        system = GestctrlSys()
        system.run()

    except Exception as e:
        print(f"Failed to start Gesture Control System: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()