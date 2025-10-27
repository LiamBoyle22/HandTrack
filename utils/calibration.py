#calibration for visualization
import numpy as np
import cv2
import glob
import os

def draw_calibration_pattern(image, pattern_size=(9, 6), square_size=0.025):