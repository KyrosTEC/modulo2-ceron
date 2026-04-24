import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


TEMPLATE_DIR = "../templates"

TEMPLATE_FILES = {
    "alto": "stop.png",
    "trabajadores": "worker.png",
    "derecha": "right.png",
    "izquierda": "left.png",
    "derecho": "straight.png",
}


def load_templates(size=(120, 120)):
    templates = {}

    for label, filename in TEMPLATE_FILES.items():
        path = os.path.join(TEMPLATE_DIR, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"No se pudo cargar template: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, size)
        templates[label] = gray

    return templates


def classify_roi(roi, templates, allowed_labels=None, size=(120, 120)):
    if roi is None or roi.size == 0:
        return None, 0, 0

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, size)

    best_label = None
    best_tm_score = -1
    best_ssim_score = -1
    best_final_score = -1

    for label, template in templates.items():
        if allowed_labels is not None and label not in allowed_labels:
            continue

        # Asegurar que el template tenga el mismo tamaño que la ROI
        template_resized = cv2.resize(template, size)

        tm_result = cv2.matchTemplate(
            roi_gray,
            template_resized,
            cv2.TM_CCOEFF_NORMED
        )

        tm_score = tm_result[0][0]

        ssim_score = ssim(
            roi_gray,
            template_resized,
            data_range=255
        )

        final_score = (tm_score * 0.6) + (ssim_score * 0.4)

        if final_score > best_final_score:
            best_label = label
            best_tm_score = tm_score
            best_ssim_score = ssim_score
            best_final_score = final_score

    return best_label, best_tm_score, best_ssim_score