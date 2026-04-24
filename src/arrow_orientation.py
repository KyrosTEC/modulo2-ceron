import cv2
import numpy as np


def classify_arrow_direction(roi):
    """
    Clasifica la flecha azul como:
    - izquierda
    - derecha
    - derecho

    usando la distribución de pixeles blancos dentro de la señal.
    """

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Detectar partes blancas de la flecha
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 80, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    h, w = mask.shape

    left = mask[:, :w // 3]
    center = mask[:, w // 3: 2 * w // 3]
    right = mask[:, 2 * w // 3:]

    top = mask[:h // 3, :]
    middle = mask[h // 3: 2 * h // 3, :]
    bottom = mask[2 * h // 3:, :]

    left_score = cv2.countNonZero(left)
    center_score = cv2.countNonZero(center)
    right_score = cv2.countNonZero(right)

    top_score = cv2.countNonZero(top)
    middle_score = cv2.countNonZero(middle)
    bottom_score = cv2.countNonZero(bottom)

    # Si la parte blanca domina arriba, probablemente es flecha recta
    if top_score > middle_score * 0.8 and center_score > left_score * 0.8 and center_score > right_score * 0.8:
        return "derecho"

    # Flecha izquierda: más pixeles blancos hacia la izquierda
    if left_score > right_score * 1.15:
        return "izquierda"

    # Flecha derecha: más pixeles blancos hacia la derecha
    if right_score > left_score * 1.15:
        return "derecha"

    return None