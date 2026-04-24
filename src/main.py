"""
Pipeline de detección de señalamientos:

1. Se lee el video frame por frame usando OpenCV.
2. Cada frame se convierte a HSV para segmentar colores relevantes:
   rojo para señales de ALTO/trabajadores y azul para señales de dirección.
3. Se aplican operaciones morfológicas para limpiar las máscaras de color.
4. Se buscan contornos en las máscaras para obtener regiones candidatas (ROI).
5. Las ROI rojas se clasifican por forma:
   - forma octagonal/circular -> ALTO
   - forma triangular -> trabajadores
6. Las ROI azules se comparan con templates para distinguir derecha,
   izquierda o seguir derecho.
7. Cada ROI se valida con Template Matching y SSIM como métricas de similitud.
8. Finalmente se dibujan las detecciones sobre el frame y se guarda un video
   de salida con los resultados.
"""

import os
import cv2
from candidate_detection import detect_candidates
from template_matching import load_templates, classify_roi
from arrow_orientation import classify_arrow_direction


VIDEO_PATH = "../videos/pista.mp4"
OUTPUT_PATH = "../output/detecciones.mp4"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    os.makedirs("../output", exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    templates = load_templates()

    detections_count = {
        "alto": 0,
        "derecha": 0,
        "izquierda": 0,
        "derecho": 0,
        "trabajadores": 0
    }

    last_detections = []
    MAX_MISSED_FRAMES = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        candidates, debug_mask_red, debug_mask_blue, debug_mask_shape = detect_candidates(frame)
        vis = frame.copy()

        for (x, y, w, h, color_label) in candidates:
            roi = frame[y:y + h, x:x + w]

            forced_label = None

            if color_label == "red_stop":
                allowed_labels = ["alto"]
                forced_label = "alto"
            elif color_label == "red_triangle":
                allowed_labels = ["trabajadores"]
                forced_label = "trabajadores"
            elif color_label == "red":
                allowed_labels = ["alto", "trabajadores"]
            elif color_label == "blue":
                allowed_labels = ["derecha", "izquierda", "derecho"]
            else:
                allowed_labels = None

            label, tm_score, ssim_score = classify_roi(
                roi,
                templates,
                allowed_labels=allowed_labels
            )

            if forced_label is not None:
                label = forced_label

            if color_label == "blue":
                arrow_label = classify_arrow_direction(roi)
                if arrow_label is not None:
                    label = arrow_label

            if label is None:
                continue

            if color_label in ["red_triangle", "red_stop"]:
                pass
            elif color_label == "blue":
                if tm_score < 0.10:
                    continue
            else:
                if tm_score < 0.20:
                    continue

            if label in detections_count:
                detections_count[label] += 1

            last_detections.append({
                "label": label,
                "box": (x, y, w, h),
                "life": MAX_MISSED_FRAMES
            })

            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = f"{label} TM:{tm_score:.2f} SSIM:{ssim_score:.2f}"
            cv2.putText(
                vis,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        new_last_detections = []

        for det in last_detections:
            det["life"] -= 1

            if det["life"] <= 0:
                continue

            det_label = det["label"]
            dx, dy, dw, dh = det["box"]

            cv2.rectangle(vis, (dx, dy), (dx + dw, dy + dh), (0, 180, 255), 2)
            cv2.putText(
                vis,
                f"{det_label} tracking",
                (dx, dy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 180, 255),
                2
            )

            new_last_detections.append(det)

        last_detections = new_last_detections

        out.write(vis)

        cv2.imshow("Deteccion", vis)
        cv2.imshow("Mask Red", debug_mask_red)
        cv2.imshow("Mask Blue", debug_mask_blue)
        cv2.imshow("Mask Red Shape", debug_mask_shape)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    print("Resumen de detecciones:")
    for label, count in detections_count.items():
        print(f"{label}: {count}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()