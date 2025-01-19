import cv2
import numpy as np

def setup_camera(camera_num=0):
    cam = cv2.VideoCapture(camera_num)

    if not cam.isOpened():
        raise IOError("Could not open camera")
    
    return cam

def get_frame(cam):
    ret, frame = cam.read()

    if not ret:
        raise IOError("Could not read camera data")
    
    return frame