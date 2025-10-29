import pyautogui
import time

class CursorController:

    def __init__(self, screen_width = None, screen_height = None):

        if screen_width is None or screen_height is None:
            screen_size = pyautogui.size()
            self.screen_width = screen_size.width
            self.screen_height = screen_size.height

        else:
            self.screen_width = screen_width
            self.screen_height = screen_height

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0

        self.is_dragging = False
        self.last_click_time = 0
        self.click_cooldown = 0.3

    def move_to(self, x, y, duration=0.0):
        
        try:
            x = max(0, min(self.screen_width - 1, int(x)))
            y = max(0, min(self.screen_height - 1, int(y)))

            pyautogui.moveTo(x, y, duration=duration)

        except Exception as e:
            print(f"Error moving cursor: {e}")
    
    def move_relative(self, dx, dy):

        try:
            pyautogui.moveRel(int(dx), int(dy))
        except Exception as e:
            print(f"Error moving cursor relatively: {e}")

    def get_position(self):
        
        pos = pyautogui.position()
        return (pos.x, pos.y)
    
    def click(self, button='left', clicks = 1):

        current_time = time.time()
        if current_time - self.last_click_time < self.click_cooldown:
            return
        
        try:
            pyautogui.click(button=button, clicks=clicks)
            self.last_click_time = current_time
        except Exception as e:
            print(f"Error clicking mouse: {e}")

    def left_click(self):
        self.click(button = 'left', clicks = 1)

    def right_click(self):
        self.click(button = 'right', clicks = 1)

    def double_click(self):
        self.click(button = 'left', clicks = 2)

    def mouse_down(self, button='left'):

        try:
            pyautogui.mouseDown(button=button)
            self.is_dragging = True
        except Exception as e:
            print(f"Error pressing mouse button down: {e}")

    def mouse_up(self, button='left'):
        try:
            pyautogui.mouseUp(button=button)
            self.is_dragging = False
        except Exception as e:
            print(f"Error releasing mouse button: {e}")

    def drag_to(self, x, y, duration = 0.2):

        try:
            x = max(0, min(self.screen_width - 1, int(x)))
            y = max(0, min(self.screen_height - 1, int(y)))

            pyautogui.dragTo(x, y, duration=duration, button='left')
        except Exception as e:
            print(f"Error dragging mouse to position: {e}")

    def scroll(self, ammount):
        
        try:
            pyautogui.scroll(int(ammount))
        except Exception as e:
            print(f"Error scrolling mouse: {e}")

    def scroll_up(self, clicks = 3):
        self.scroll(clicks)

    def scroll_down(self, clicks = 3):
        self.scroll(-clicks)

    def set_click_cooldown(self, cooldown):
        self.click_cooldown = cooldown

    def map_coordinates(self, x, y, source_width, source_height):
        x_norm = x / source_width
        y_norm = y / source_height

        screen_x = int(x_norm * self.screen_width)
        screen_y = int(y_norm * self.screen_height)

        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))

        return (screen_x, screen_y)