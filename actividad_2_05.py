import time
import sys
import math
import numpy as np
import cv2


class TrafficLightDetection:
    def __init__(self):
        self.cameraWidth  = 320
        self.cameraHeight = 240

        # Rangos HSV calibrados con valores reales del semáforo
        #   Verde:    H=54  S=112  V=196
        #   Amarillo: H=31  S=120  V=199
        #   Rojo:     H=176 S=126  V=176
        self.color_ranges = {
            "green": [
                (np.array([46,  72,  146]), np.array([62,  152, 246])),
            ],
            "yellow": [
                (np.array([23,  80,  149]), np.array([39,  160, 249])),
            ],
            "red": [
                (np.array([0, 86,  126]), np.array([10, 166, 226])),
            ],
        }

    def detect_state(self, image):
        """
        Detecta el estado del semáforo y lo reporta como texto.
        :param image: Imagen BGR capturada por la cámara (numpy array).
        :return: String con alguno de los siguientes contenidos: green, yellow, red, none
        """
        state = "none"

        if image is None:
            return state

        # 1. Suavizar para reducir ruido
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # 2. Convertir a HSV
        roi = image[0:int(image.shape[0]/2), 0:int(image.shape[1]*0.5)]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)

        best_score = 0

        for color, ranges in self.color_ranges.items():
            # 3. Crear máscara para este color
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

            # 4. Limpiar ruido con morfología
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 5. Evaluar contornos: área × circularidad (premia formas redondas)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            score = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 8:
                    continue
                p    = cv2.arcLength(cnt, True)
                circ = 4 * np.pi * area / (p ** 2) if p > 0 else 0
                score += area * (1 + circ)

            # 6. El color con mayor puntaje gana
            if score > best_score:
                best_score = score
                state = color

        # Umbral mínimo para evitar falsos positivos
        if best_score < 20:
            state = "none"

        return state