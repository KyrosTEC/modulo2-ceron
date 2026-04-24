import cv2
import numpy as np
import os


class TrafficSignDetection:
    def __init__(self, templates_path=""):

        self.templates = {}
        self.last_detections = []  

        file_mapping = {
            "Stop": "stop.png",
            "Workers": "worker.png",
            "Go Straight": "straight.png",
            "Turn Left": "left.png",
            "Turn Right": "right.png"
        }

        for name, filename in file_mapping.items():
            path = os.path.join(templates_path, filename)
            img = cv2.imread(path, 0)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                self.templates[name] = img
            else:
                print(f"Error cargando {filename}")

        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([15, 255, 255])
        self.lower_red2 = np.array([160, 50, 50])
        self.upper_red2 = np.array([180,255,255])

        self.lower_yellow = np.array([15,70,70])
        self.upper_yellow = np.array([40,255,255])

        self.lower_blue = np.array([90,80,80])
        self.upper_blue = np.array([140,255,255])

    def preprocess(self, img):
        img = cv2.equalizeHist(img)
        img = cv2.GaussianBlur(img, (5,5), 0)
        return img

    def edge_density(self, img):
        edges = cv2.Canny(img, 50, 150)
        return np.sum(edges) / (img.shape[0] * img.shape[1])

    def red_ratio(self, mask_red_roi, w, h):
        return cv2.countNonZero(mask_red_roi) / float(w * h)

    def strong_red(self, roi):
        b, g, r = cv2.split(roi)
        return np.mean((r > g + 20) & (r > b + 20))

    def brightness(self, roi):
        return np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    def detect_sign(self, frame):

        if frame is None:
            return "none", 0, None

        height, width = frame.shape[:2]

        frame = frame[:, width//2:]
        offset_x = width // 2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_red = cv2.inRange(hsv, self.lower_red1, self.upper_red1) + \
                   cv2.inRange(hsv, self.lower_red2, self.upper_red2)

        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_blue   = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        mask = mask_red + mask_yellow + mask_blue

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for cnt in contours:

            area = cv2.contourArea(cnt)
            if area < 80:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if w < 10 or h < 10:
                continue

            roi = frame[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.resize(roi_gray, (64, 64))
            roi_gray = self.preprocess(roi_gray)

            roi_hsv = hsv[y:y+h, x:x+w]

            if np.mean(roi_hsv[:,:,1]) < 40:
                continue

            red = cv2.countNonZero(mask_red[y:y+h, x:x+w])
            yellow = cv2.countNonZero(mask_yellow[y:y+h, x:x+w])
            blue = cv2.countNonZero(mask_blue[y:y+h, x:x+w])

            # 🔴 STOP 
            if red > yellow and red > blue:

                aspect_ratio = w / float(h)
                if aspect_ratio < 0.75 or aspect_ratio > 1.25:
                    continue

                red_ratio = self.red_ratio(mask_red[y:y+h, x:x+w], w, h)
                if red_ratio < 0.30:
                    continue

                if self.strong_red(roi) < 0.4:
                    continue

                if self.brightness(roi) < 60:
                    continue

                cx1, cy1 = int(w*0.25), int(h*0.25)
                cx2, cy2 = int(w*0.75), int(h*0.75)
                center = mask_red[y+cy1:y+cy2, x+cx1:x+cx2]

                if cv2.countNonZero(center) < 0.5 * center.size:
                    continue

                if self.edge_density(roi_gray) < 6:
                    continue

                candidates = ["Stop"]

            elif yellow > red and yellow > blue:
                candidates = ["Workers"]

            elif blue > red and blue > yellow:
                candidates = ["Go Straight", "Turn Left", "Turn Right"]

            else:
                continue

            # TEMPLATE MATCHING
            for label in candidates:

                template = self.templates[label]
                template = self.preprocess(template)

                scales = [0.7, 1.0, 1.3]

                for scale in scales:

                    size = int(64 * scale)
                    if size < 16:
                        continue

                    temp_resized = cv2.resize(template, (size, size))

                    try:
                        res = cv2.matchTemplate(
                            roi_gray,
                            temp_resized,
                            cv2.TM_CCOEFF_NORMED
                        )

                        _, score, _, _ = cv2.minMaxLoc(res)

                        thr = 0.35 if label == "Stop" else 0.25

                        if score > thr:
                            detections.append({
                                "label": label,
                                "score": score,
                                "box": (
                                    (x + offset_x, y),
                                    (x + offset_x + w, y + h)
                                )
                            })

                    except:
                        continue

        self.last_detections = detections

        if len(detections) == 0:
            return "none", 0, None

        best = max(detections, key=lambda d: d["score"])
        return best["label"], best["score"], best["box"]