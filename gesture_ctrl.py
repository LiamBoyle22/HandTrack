#MAIN
#use this to run the application
import cv2 
import mediapipe as mp
import pyautogui
import tkinter as tk
from tkinter import ttk
import threading
import numpy as np

#MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

#screen Size
screen_width, screen_height = pyautogui.size()

#traacking state
is_running = False
capture = None

#Smoothening parameters for cursor movement
prev_x, prev_y = 0, 0
smoothening = 0.5

def main():
    global is_running, capture, prev_x, prev_y

    is_running = True
    capture = cv2.VideoCapture(0)

    #set camera resolution
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while is_running:
        