import cv2
from candidate_detection import detect_candidates


VIDEO_PATH = "../videos/pista.mp4"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        candidates, debug_mask_red, debug_mask_blue = detect_candidates(frame)

        vis = frame.copy()

        for (x, y, w, h, label) in candidates:
            color = (0, 255, 0) if label == "red" else (255, 0, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                vis,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("Frame", vis)
        cv2.imshow("Mask Red", debug_mask_red)
        cv2.imshow("Mask Blue", debug_mask_blue)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()