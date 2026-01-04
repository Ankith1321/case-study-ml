# Road Lane Detection – Case Study (UNet, Hough Transform, YOLO)

## Overview
This project is a comparative case study on **road lane detection** using both **deep learning** and **classical computer vision** approaches.  
The main objective is not only to achieve lane detection, but to **analyze, compare, and justify** different methods based on accuracy, robustness, and computational cost.

Three approaches were implemented and evaluated:
1. **UNet-based semantic segmentation**
2. **UNet with Hough Transform post-processing**
3. **YOLO-based segmentation**

The project focuses on understanding *why* certain approaches perform better for lane detection rather than only reporting numerical accuracy.

---

## Dataset
The experiments are conducted using the **TuSimple Lane Detection Dataset**, which contains real-world highway driving images annotated with lane markings.

Key characteristics of the dataset:
- Thin and continuous lane markings
- Highway driving scenarios
- Challenging lighting conditions, shadows, curves, and worn-out lanes
- Suitable for pixel-wise segmentation and geometric lane detection

Due to size constraints, the raw dataset is not included in this repository.

---

## Project Structure

road_lane_detection/
│
├── src/ # Training, evaluation, inference code
│ ├── train.py
│ ├── evaluate.py
│ ├── lane_detection.py
│ ├── dataset.py
│ ├── models.py
│ └── main.ipynb
│
├── outputs/ # Saved predictions and plots
│ ├── report_figures_unet/
│ ├── report_figures_yolo/
│ └── comparison_outputs/
│
├── runs/ # YOLO training logs
├── models/ # Model checkpoints
├── requirements.txt
├── README.md


---


## Methodology

### 1. UNet – Semantic Segmentation
A lightweight **UNet architecture** was trained to perform **pixel-wise lane segmentation**.

- Input: RGB road images
- Output: Binary lane mask
- Loss Function: **Binary Cross Entropy + Dice Loss**
- Class imbalance handled using **positive class weighting**
- Evaluation Metrics:
  - Dice Score
  - Intersection over Union (IoU)
  - Precision, Recall, F1-score

**Observation:**  
UNet learns lane regions effectively but produces noisy masks due to the thin nature of lane markings.

---

### 2. UNet + Hough Transform (Hybrid Approach)
To improve lane structure and continuity, **Hough Line Transform** was applied as a post-processing step on UNet predictions.

**Pipeline:**
1. UNet probability output
2. Thresholding and morphological operations
3. Edge detection
4. Region of Interest (ROI) masking
5. Hough line detection
6. Averaging left and right lane lines

**Why Hough Transform?**
- Enforces geometric constraints
- Converts noisy pixel masks into continuous lane lines
- Improves visual interpretability and structural consistency

**Result:**  
This hybrid approach produced the **most stable and visually consistent lane detection results**.

---

### 3. YOLO Segmentation
A YOLO-based segmentation model was trained for **instance-level lane detection**.

- Faster inference compared to UNet
- Uses object-level segmentation masks
- Evaluation Metrics:
  - Mask mAP@50
  - Mask mAP@50–95
  - Precision and Recall

**Limitation:**  
YOLO struggles with thin, elongated lane structures and lacks geometric continuity, making it less suitable for precise lane detection.

---

## Results Summary

### Quantitative Results (Best Epoch)
| Model | Primary Metric | Best Value |
|------|---------------|-----------|
| UNet | Dice Score | ~0.52 |
| YOLO-Seg | Mask mAP@50–95 | ~0.20 |

### Qualitative Observations
- UNet alone provides correct lane regions but noisy boundaries
- UNet + Hough produces smooth and continuous lane lines
- YOLO detects lane regions but lacks geometric consistency

### Training Analysis
- UNet converges faster with fewer epochs
- YOLO requires more epochs and higher computational cost
- Training loss and validation curves are included in the `outputs/` directory

---

## Final Conclusion
Although YOLO is a powerful and general-purpose segmentation model, it is **not optimal for lane detection**, where thin, continuous, and structured features are critical.

The **UNet + Hough Transform hybrid approach**:
- Provides better structural accuracy
- Is computationally efficient
- Produces visually interpretable lane lines

**Final Model Selection:**  
✅ **UNet with Hough Transform post-processing**

This choice is justified both quantitatively and qualitatively.

---

## How to Run the Project

### Install Dependencies
```bash
pip install -r requirements.txt


Train UNet
python -m src.train

Evaluate UNet
python -m src.evaluate

Run Lane Detection
python -m src.main

