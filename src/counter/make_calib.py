"""Extract INT8 calibration frames and write the dataset yaml that ultralytics' TensorRT export reads.

Calibration only records the value ranges of the network's tensors, but it must still see images like the ones
deployed, and none of the frames we evaluate on: so the E18 frames stop at 24 s (evaluation starts at 25 s),
the A40 is left out entirely, and the two Chicago drone videos add variety.
"""
from pathlib import Path

import cv2
from ultralytics import YOLO

V = Path("data/videos")
SOURCES = [  # (video, start s, end s, frames to take, evenly spaced)
    (V / "E18_nord_for_Ramsum,_Vestfold.webm", 0, 24, 240),
    (V / "Lake_Shore_Drive_Traffic.webm", 0, None, 130),
    (V / "Jane_M._Byrne_Interchange_Traffic.webm", 0, None, 130),
]
OUT = Path("data/calib")


def main():
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for video, start, end, n in SOURCES:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {video}")
        fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        first, last = round(start * fps), min(total, round(end * fps)) if end is not None else total
        wanted = {first + k * (last - first) // n for k in range(n)}
        for i in range(last):  # sequential grab: seeking in compressed video can land on the wrong frame
            if not cap.grab():
                break
            if i in wanted:
                cv2.imwrite(str(img_dir / f"{video.stem[:12]}_{i:05d}.jpg"), cap.retrieve()[1],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
        cap.release()
    names = YOLO("yolo11n.pt").names  # same 80 COCO classes as the model; no labels needed for calibration
    yaml = f"path: {OUT.resolve().as_posix()}\ntrain: images\nval: images\nnames:\n"
    yaml += "".join(f"  {k}: {v}\n" for k, v in names.items())
    (OUT / "calib.yaml").write_text(yaml)
    count = len(list(img_dir.glob("*.jpg")))
    assert count >= 300, f"only {count} calibration images; TensorRT recommends ~500"
    print(f"{count} images in {img_dir}, dataset yaml {OUT / 'calib.yaml'}")


if __name__ == "__main__":
    main()
