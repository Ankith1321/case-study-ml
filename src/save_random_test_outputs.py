import random
from pathlib import Path
import cv2

from src.lane_detection import detect_lanes


def main():
    # ---------------- CONFIG ----------------
    test_root = Path(r"data/raw/tusimple/test_set/clips")

    # OLD results folder (before improvements)
    before_dir = Path("outputs/random_test_results")

    # NEW results folder (after improvements)
    out_dir = Path("outputs/random_test_results_after")
    out_dir.mkdir(parents=True, exist_ok=True)

    N = 20
    THR = 0.30

    # ---------------- COLLECT ALL TEST IMAGES ----------------
    all_imgs = list(test_root.rglob("*.jpg"))
    if not all_imgs:
        raise FileNotFoundError(f"No .jpg files found under: {test_root.resolve()}")

    # ---------------- REUSE OR SAMPLE ----------------
    prev_list = before_dir / "selected_files.txt"

    if prev_list.exists():
        # Reuse exact same images (fair comparison)
        picks = [
            Path(p.strip())
            for p in prev_list.read_text(encoding="utf-8").splitlines()
            if p.strip()
        ]
        print(f"[INFO] Reusing {len(picks)} images from previous run.")
    else:
        # Randomly sample new images
        N = min(N, len(all_imgs))
        picks = random.sample(all_imgs, N)
        print(f"[INFO] Sampling {N} new random test images.")

    # Save selected list for traceability
    (out_dir / "selected_files.txt").write_text(
        "\n".join(str(p.resolve()) for p in picks),
        encoding="utf-8"
    )

    # ---------------- INFERENCE + SAVE ----------------
    for i, img_path in enumerate(picks, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read: {img_path}")
            continue

        out = detect_lanes(img, thr=None)

       # Robust filename creation (works for absolute paths)
        parts = img_path.parts[-3:]  # keep last 3 folders + filename
        safe_name = "__".join(parts)
        save_path = out_dir / f"{i:02d}__{safe_name}"


        cv2.imwrite(str(save_path), out)
        print(f"[{i:02d}/{len(picks)}] Saved: {save_path}")

    print("\n[DONE] Results saved to:")
    print(out_dir.resolve())
    print("[DONE] Selected file list saved to:")
    print((out_dir / "selected_files.txt").resolve())


if __name__ == "__main__":
    main()
