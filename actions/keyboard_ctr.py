import pyautogui 
import time
import platform

class KeyBoardController:

    def __init__(self):
        self.os_name = platform.system()

        if self.os_name == "Darwin":  # macOS
            self.modifier = "command"
        else:  # Windows and Linux
            self.modifier = "ctrl"
        
        self.last_action_time = 0
        self.action_cooldown = 0.5

    def check_cooldown(self):
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            return False
        self.last_action_time = current_time
        return True
    
    def press_key(self, key):

        try:
            pyautogui.press(key)
        except Exception as e:
            print(f"Error pressing key {key}: {e}")

    def hotkey(self, *keys):