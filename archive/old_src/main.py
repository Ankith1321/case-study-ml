import cv2
from pathlib import Path
from src.lane_detection import detect_lanes

folder = Path(r"E:\dl_cv_mini_project\road_lane_detection\data\raw\tusimple\test_set\clips\0530\1492626047222176976_0")

imgs = sorted(folder.glob("*.jpg"))
if not imgs:
    raise FileNotFoundError(f"No .jpg found in: {folder}")

for p in imgs:
    img = cv2.imread(str(p))
    if img is None:
        continue

    out = detect_lanes(img, thr=0.35)
    cv2.imshow("Lane Detection (UNet + Hough)", out)

    # press 'q' to quit, any other key to go next
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
