"""Stage 1: run YOLO on every frame of a video, save an annotated copy, report FPS."""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

VEHICLES = [2, 3, 5, 7]  # COCO class ids: car, motorcycle, bus, truck
WARMUP = 10  # first frames are slow (CUDA init, kernel selection): excluded from timing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", default="runs/stage1/out.mp4")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    model = YOLO(a.model)

    frame_times, speeds, n_dets = [], [], []
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()  # frame: numpy array (h, w, 3), uint8, BGR
        if not ok:
            break
        r = model(frame, imgsz=a.imgsz, conf=a.conf, classes=VEHICLES, device=0, verbose=False)[0]
        writer.write(r.plot())  # plot() draws boxes + labels on a copy of the frame
        frame_times.append(time.perf_counter() - t0)
        speeds.append(r.speed)  # ms: preprocess / inference / postprocess
        n_dets.append(len(r.boxes))

    cap.release()
    writer.release()

    t, s = frame_times[WARMUP:], speeds[WARMUP:]
    if not t:
        raise SystemExit("video too short to measure")
    print(f"frames: {len(frame_times)}  model: {a.model}  imgsz: {a.imgsz}  conf: {a.conf}")
    print(f"end-to-end: {len(t) / sum(t):.1f} FPS  ({1000 * sum(t) / len(t):.1f} ms/frame)")
    for k in ("preprocess", "inference", "postprocess"):
        print(f"  {k:<12} {sum(x[k] for x in s) / len(s):.1f} ms")
    print(f"vehicles per frame: mean {sum(n_dets) / len(n_dets):.1f}, max {max(n_dets)}")
    print(f"saved: {a.out}")


if __name__ == "__main__":
    main()
