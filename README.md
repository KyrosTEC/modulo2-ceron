# Computer Vision for Robotics Module

## Overview

This repository contains activities developed for the **Computer Vision for Robotics module**, focused on designing perception pipelines for robotic systems using camera-based inputs.

The project emphasizes classical computer vision techniques to detect and classify traffic signs in a simulated driving environment.

---

## Activity: Traffic Sign Detection

The objective of this activity is to detect and classify the following traffic signs from a video stream:

- Stop
- Workers
- Go Straight
- Turn Left
- Turn Right

The system processes a rendered POV driving video and outputs a labeled video with detected signs.

---

## Detection Pipeline

The traffic sign detection system is based on a hybrid pipeline that combines **color segmentation, geometric filtering, and template-based classification**, enhanced with temporal smoothing.

### Pipeline Steps

1. **Frame Acquisition**  
   The input video is processed frame by frame using OpenCV.

2. **Color Space Transformation**  
   Each frame is converted to the HSV color space to improve robustness in color detection.

3. **Color Segmentation**  
   Masks are generated to isolate relevant regions:
   - **Red mask** → Stop and Workers signs  
   - **Blue mask** → Directional signs (left, right, straight)

4. **Morphological Processing**  
   Morphological operations (closing and dilation) are applied to reduce noise and improve region continuity.

5. **Candidate Extraction (ROI)**  
   Contours are extracted from the masks, and candidate regions are filtered based on:
   - Area
   - Aspect ratio
   - Shape approximation

6. **Shape-based Classification (Red Signs)**  
   Red regions are classified using polygon approximation:
   - **Octagonal / circular shape → Stop sign**
   - **Triangular shape → Workers sign**

7. **Template Matching (Blue Signs)**  
   Blue regions are classified by comparing each ROI against predefined templates:
   - Left
   - Right
   - Straight

8. **Similarity Validation**  
   Each candidate is evaluated using:
   - Template Matching score
   - SSIM (Structural Similarity Index)

9. **Directional Refinement**  
   For directional signs, pixel distribution analysis is used to refine classification.

10. **Temporal Smoothing (Tracking)**  
   Detected signs are stored temporarily to maintain stable detection across frames.

11. **Visualization & Output**  
   Bounding boxes and labels are drawn on each frame, and a processed video is generated.

---

## Methodology

This implementation focuses on classical computer vision techniques:

- Color segmentation (HSV)
- Shape analysis (contours + polygon approximation)
- Template Matching
- SSIM similarity metric

This approach provides a lightweight and interpretable alternative to deep learning-based models.

---

## Technologies Used

- Python
- OpenCV
- NumPy
- scikit-image

---

## How to Run

1. Install dependencies:

```bash
pip install opencv-python numpy scikit-image
```

2. Run the main script:

```bash
python main.py
```

3. Output video will be saved in:

```bash
/output/detecciones.mp4
```

---

## Project Structure

```
project/
│── src/
│   ├── main.py
│   ├── candidate_detection.py
│   ├── template_matching.py
│   ├── arrow_orientation.py
│
│── templates/
│   ├── stop.png
│   ├── worker.png
│   ├── left.png
│   ├── right.png
│   ├── straight.png
│
│── videos/
│   └── pista.mp4
│
│── output/
│   └── detecciones.mp4
```

---

## Results

The system is capable of:

- Detecting traffic signs in real-time
- Classifying signs using hybrid techniques
- Maintaining stable detections through temporal smoothing

---

## Notes

- The system works best under controlled lighting conditions.
- Performance may degrade with extreme blur or occlusions.
- Future improvements could include integrating deep learning models such as YOLO.
