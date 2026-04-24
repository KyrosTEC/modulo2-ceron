import time 
import sys
import math
import numpy as np
import cv2

class CenterLineDetector:
    def __init__(self):
        self.cameraWidth = 320
        self.cameraHeight = 240
        self.last_x = None
        self.prev_x = None
        self.smooth_x = None
        self.alpha = 0.3

    def detect_center_line(self, image):
        best_candidate = (0, 0)
        h, w, _ = image.shape

        roi = image[int(3 * h / 4):h, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        center_x = w // 2
        best_score = float('inf')

        if self.last_x is None:
            reference_x = center_x
        else:
            if self.prev_x is not None:
                delta = self.last_x - self.prev_x
                delta = max(min(delta, 20), -20)
                reference_x = int(self.last_x + delta)
            else:
                reference_x = self.last_x

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            bottom_point = tuple(cnt[cnt[:, :, 1].argmax()][0])
            cx_bottom = bottom_point[0]

            dist_center = abs(cx - center_x)
            height_weight = -cv2.boundingRect(cnt)[3] 
            area_weight = -area / 100.0
            score = dist_center + height_weight + area_weight

            if self.last_x is not None:
                motion = abs(self.last_x - self.prev_x) if self.prev_x is not None else 0
                jump_weight = 0.8 if motion > 25 else 2.0
                jump = abs(cx - reference_x)
                score += jump * jump_weight

            curve_bias = abs(cx_bottom - center_x)
            score += curve_bias * 2.0

            if self.last_x is None and abs(cx - center_x) > 50:
                continue

            if cv2.boundingRect(cnt)[2] > 50: 
                score += 200

            if score < best_score:
                best_score = score
                best_candidate = (cx, cy + int(3 * h / 4))

        if best_candidate == (0, 0):
            final_x = self.last_x if self.last_x is not None else center_x
        else:
            self.prev_x = self.last_x
            self.last_x = best_candidate[0]
            final_x = best_candidate[0]

        return (final_x, best_candidate[1] if best_candidate != (0,0) else int(0.9 * h))