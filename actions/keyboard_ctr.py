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

        if not self.check_cooldown():
            return
        
        try:
            pyautogui.hotkey(*keys)
            shortcut_name = '+'.join(keys)
            print(f"Pressed hotkey: {shortcut_name}")
        except Exception as e:
            print(f"Error pressing hotkey {keys}: {e}")

    def type_text(self, text, interval = 0.05):

        try:
            pyautogui.write(text, interval=interval)
        except Exception as e:
            print(f"Error typing text '{text}': {e}")

    def copy(self):
        self.hotkey(self.modifier, 'c')

    def paste(self):
        self.hotkey(self.modifier, 'v')

    def cut(self):
        self.hotkey(self.modifier, 'x')

    def undo(self):
        self.hotkey(self.modifier, 'z')

    def redo(self):
        
        if self.os_name == "Darwin": # macOS
            self.hotkey(self.modifier, 'shift', 'z')
        else:  # Windows and Linux
            self.hotkey(self.modifier, 'y')

    def select_all(self):
        self.hotkey(self.modifier, 'a')

    def backspace(self):
            self.press_key('backspace')

    def enter(self):
        self.press_key('enter')

    def tab(self):
        self.press_key('tab')

    def escape(self):
        self.press_key('esc')

    def minimize_window(self):
        if self.os_name == "Darwin":  # macOS
            self.hotkey('command', 'm')
        else:
            self.hotkey('win', 'down')