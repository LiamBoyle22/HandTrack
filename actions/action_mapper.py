#mapping configurations for actions
import pyautogui
from typing import Any, Callable, Dict
from actions.cursor_ctrl import CursorController
from actions.keyboard_ctr import KeyBoardController

class ActionMapper:
    def __init__(self, cursor: CursorController | None = None,
                   keyboard: KeyBoardController | None = None):
            self.cursor = cursor or CursorController()
            self.keyboard = keyboard or KeyBoardController()

            self.cursor_action_map: Dict[str, Callable[..., Any]] = {
                "move_to": self.cursor.move_to,
                "move_relative": self.cursor.move_relative,
                "get_position": self.cursor.get_position,
                "click": self.cursor.click,
                "left_click": self.cursor.left_click,
                "right_click": self.cursor.right_click,
                "double_click": self.cursor.double_click,
                "mouse_down": self.cursor.mouse_down,
                "mouse_up": self.cursor.mouse_up,
                "drag_to": self.cursor.drag_to,
                "scroll_up": self.cursor.scroll_up,
                "scroll_down": self.cursor.scroll_down,
                "SET_CLICK_COOLDOWN": self.cursor.set_click_cooldown,
                "MAP_COORDINATES": self.cursor.map_coordinates,
            }

            self.keyboard_action_map: Dict[str, Callable[..., Any]] = {
                  "key_press": self.keyboard.press_key,
                  "type_text": self.keyboard.type_text,
                  "copy": self.keyboard.copy,
                  "paste": self.keyboard.paste,
                  "cut": self.keyboard.cut,
                  "undo": self.keyboard.undo,
                  "redo": self.keyboard.redo,
                  "select_all": self.keyboard.select_all,
                  "backspace": self.keyboard.backspace,
                  "enter": self.keyboard.enter,
                  "tab": self.keyboard.tab,
                  "escape": self.keyboard.escape,
                  "minimize_window": self.keyboard.minimize_window,
            }

    def ping_action(self, action: str, *args, **kwargs):
        if action in self.cursor_action_map:
            return self.cursor_action_map[action](*args, **kwargs)
        if action in self.keyboard_action_map:
            return self.keyboard_action_map[action](*args, **kwargs)
        raise ValueError(f"Unknown action: {action}")