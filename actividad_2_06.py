import time
import sys
import math
import numpy as np
import cv2

class TrafficSignsDetection:
    def __init__(self):
        cameraWidth = 320
        cameraHeight = 240

    def detect_state(self, image):
        """
        Detecta el la señal de tránsito y lo reporta como texto.
        return: String con las diferentes señales de tránsito detectadas, por ejemplo: stop, workers, go straight, turn left, turn right, none
        usaremos template matching e interseccion de histogramas, tambien podemos usar mascaras.
        """
        sign = "none"
        h, w, _ = image.shape
        roi = image[0:h, 0:w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_lower1 = np.array([0, 40, 40])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([179, 40, 40])
        red_upper2 = np.array([179, 255, 255])

        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2) 
        red_count = cv2.countNonZero(mask_red)
        if red_count > 100:
            sign = "stop"
        

        return sign

