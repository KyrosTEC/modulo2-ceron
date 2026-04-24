import cv2
import numpy as np


def expand_box(x, y, w, h, frame_shape, scale=0.45):
    H, W = frame_shape[:2]

    pad_x = int(w * scale)
    pad_y = int(h * scale)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    return x1, y1, x2 - x1, y2 - y1


def filter_contours(mask, color_label, frame_shape, min_area=120):
    candidates = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < 10 or h < 10:
            continue

        aspect_ratio = w / float(h)

        if aspect_ratio < 0.45 or aspect_ratio > 2.2:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        sides = len(approx)

        area_box = w * h
        extent = area / float(area_box)

        x, y, w, h = expand_box(x, y, w, h, frame_shape, scale=0.45)

        if color_label == "red":
            # STOP: rojo + forma casi cuadrada/octagonal
            if 6 <= sides <= 12 and 0.75 <= aspect_ratio <= 1.25:
                candidates.append((x, y, w, h, "red_stop"))

            # Trabajadores: rojo + forma triangular
            elif 3 <= sides <= 5 and 0.75 <= aspect_ratio <= 1.35:
                candidates.append((x, y, w, h, "red_triangle"))

            # Fallback: si no sabemos, dejamos que template matching decida
            else:
                candidates.append((x, y, w, h, "red"))

        elif color_label == "blue":
            # Señales circulares azules
            if aspect_ratio >= 0.55 and aspect_ratio <= 1.55 and extent > 0.35:
                candidates.append((x, y, w, h, "blue"))

    return candidates


def detect_candidates(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 80, 60])
    upper_red_1 = np.array([12, 255, 255])

    lower_red_2 = np.array([165, 80, 60])
    upper_red_2 = np.array([180, 255, 255])

    # Azul
    lower_blue = np.array([85, 60, 40])
    upper_blue = np.array([135, 255, 255])

    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel_small = np.ones((3, 3), np.uint8)
    kernel_big = np.ones((5, 5), np.uint8)

    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_big)
    mask_red = cv2.dilate(mask_red, kernel_small, iterations=1)

    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel_big)
    mask_blue = cv2.dilate(mask_blue, kernel_small, iterations=1)

    candidates_red = filter_contours(mask_red, "red", frame.shape, min_area=80)
    candidates_blue = filter_contours(mask_blue, "blue", frame.shape, min_area=180)

    candidates = candidates_red + candidates_blue

    return candidates, mask_red, mask_blue, mask_red