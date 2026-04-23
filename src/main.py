import cv2
from candidate_detection import detect_candidates
from template_matching import load_templates, classify_roi


VIDEO_PATH = "../videos/pista.mp4"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    templates = load_templates()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        candidates, debug_mask_red, debug_mask_blue = detect_candidates(frame)

        vis = frame.copy()

        for (x, y, w, h, color_label) in candidates:
            roi = frame[y:y + h, x:x + w]

            label, tm_score, ssim_score = classify_roi(roi, templates)

            if label is None:
                continue

            # Umbrales iniciales
            if tm_score < 0.25 or ssim_score < 0.25:
                continue

            color = (0, 255, 0)

            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            text = f"{label} TM:{tm_score:.2f} SSIM:{ssim_score:.2f}"
            cv2.putText(
                vis,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        cv2.imshow("Deteccion", vis)
        cv2.imshow("Mask Red", debug_mask_red)
        cv2.imshow("Mask Blue", debug_mask_blue)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()