import cv2
import numpy as np
from time import sleep

def count_vehicles_in_video(video_path):
    """Counts vehicles in a video using background subtraction."""
    width_min, height_min, offset, pos_line = 80, 80, 6, 550
    detec, cars = [], 0
    cap = cv2.VideoCapture(video_path)
    subtraction = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = subtraction.apply(blur)
        dilated = cv2.dilate(img_sub, np.ones((5, 5)))
        contour, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.line(frame, (25, pos_line), (1200, pos_line), (255, 127, 0), 3)

        for c in contour:
            (x, y, w, h) = cv2.boundingRect(c)
            if w >= width_min and h >= height_min:
                centro = (x + w//2, y + h//2)
                detec.append(centro)
                for (x, y) in detec:
                    if y < (pos_line + offset) and y > (pos_line - offset):
                        cars += 1
                        detec.remove((x, y))

        cap.release()
    return cars
