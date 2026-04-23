import cv2
import numpy as np


def filter_contours(mask, color_label, min_area=500):
    candidates = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = w / float(h)
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            continue

        candidates.append((x, y, w, h, color_label))

    return candidates


def detect_candidates(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rojo en HSV suele ocupar dos rangos
    lower_red_1 = np.array([0, 100, 80])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 100, 80])
    upper_red_2 = np.array([180, 255, 255])

    # Azul
    lower_blue = np.array([90, 100, 80])
    upper_blue = np.array([130, 255, 255])

    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

    candidates_red = filter_contours(mask_red, "red")
    candidates_blue = filter_contours(mask_blue, "blue")

    candidates = candidates_red + candidates_blue

    return candidates, mask_red, mask_blue