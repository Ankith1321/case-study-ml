import cv2
import numpy as np
import torch

from src.models import UNetSmall

# -----------------------------
# Global configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_H, IMG_W = 360, 640
CKPT_PATH = "models/checkpoints/unet_tusimple_best.pt"

_MODEL = None


# -----------------------------
# Model loading
# -----------------------------
def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model = UNetSmall().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _MODEL = model
    return _MODEL


# -----------------------------
# Preprocessing
# -----------------------------
def _preprocess_frame_bgr(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
    x = (rgb.astype(np.float32) / 255.0)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    return x, rgb


# -----------------------------
# Mask post-processing
# -----------------------------
def _postprocess_mask(prob_hw, thr):
    mask = (prob_hw >= thr).astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    return mask


# -----------------------------
# ROI + Hough
# -----------------------------
def _roi_edges(edges):
    H, W = edges.shape
    roi = np.zeros_like(edges)

    poly = np.array([[
        (int(0.05 * W), H),
        (int(0.45 * W), int(0.60 * H)),
        (int(0.55 * W), int(0.60 * H)),
        (int(0.95 * W), H),
    ]], dtype=np.int32)

    cv2.fillPoly(roi, poly, 255)
    return cv2.bitwise_and(edges, roi)


def _hough_lines(edges_roi):
    return cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=25,
    )


# -----------------------------
# Lane averaging utilities
# -----------------------------
def _split_left_right(lines, W):
    left, right = [], []

    if lines is None:
        return None, None

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)

        if m < -0.3 and x1 < W * 0.7 and x2 < W * 0.7:
            left.append([x1, y1, x2, y2])
        elif m > 0.3 and x1 > W * 0.3 and x2 > W * 0.3:
            right.append([x1, y1, x2, y2])

    left = np.array(left) if len(left) else None
    right = np.array(right) if len(right) else None
    return left, right


def _average_lane_line(lines):
    if lines is None:
        return None

    ms, bs = [], []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        ms.append(m)
        bs.append(b)

    if len(ms) == 0:
        return None
    return float(np.mean(ms)), float(np.mean(bs))


def _draw_avg_line(img, mb, y_bottom, y_top, color=(0, 0, 255), thickness=5):
    if mb is None:
        return img

    m, b = mb
    if abs(m) < 1e-6:
        return img

    x_bottom = int((y_bottom - b) / m)
    x_top = int((y_top - b) / m)

    cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), color, thickness)
    return img

def _auto_threshold(prob_hw: np.ndarray, pctl: float = 97.0, lo: float = 0.20, hi: float = 0.45) -> float:
    """
    Choose a per-image threshold from the probability distribution.
    pctl=97 means keep roughly the top 3% most confident pixels.
    Clamp to [lo, hi] to avoid extreme thresholds.
    """
    t = float(np.percentile(prob_hw, pctl))
    return max(lo, min(hi, t))


# -----------------------------
# Main API
# -----------------------------
def detect_lanes(frame_bgr, thr=None):
    model = _load_model()

    x, rgb_resized = _preprocess_frame_bgr(frame_bgr)

    with torch.no_grad():
        logits = model(x.to(DEVICE))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    
    # Adaptive threshold (Part C)
    if thr is None:
     thr = _auto_threshold(prob, pctl=97.0, lo=0.20, hi=0.45)


    # First attempt
    mask = _postprocess_mask(prob, thr)
    edges = cv2.Canny(mask, 50, 150)
    edges_roi = _roi_edges(edges)
    lines = _hough_lines(edges_roi)

    # Retry once with lower threshold if needed
    if lines is None and thr > 0.20:
        mask = _postprocess_mask(prob, thr - 0.10)
        edges = cv2.Canny(mask, 50, 150)
        edges_roi = _roi_edges(edges)
        lines = _hough_lines(edges_roi)

    # Lane averaging
    H, W = rgb_resized.shape[:2]
    left_lines, right_lines = _split_left_right(lines, W)

    left_mb = _average_lane_line(left_lines)
    right_mb = _average_lane_line(right_lines)

    out_bgr = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)

    y_bottom = H - 1
    y_top = int(H * 0.55)


    out_bgr = _draw_avg_line(out_bgr, left_mb, y_bottom, y_top)
    out_bgr = _draw_avg_line(out_bgr, right_mb, y_bottom, y_top)

    return out_bgr
