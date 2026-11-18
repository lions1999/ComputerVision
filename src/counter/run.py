"""Detect, track and count vehicles crossing a line; save an annotated video and a CSV of crossings."""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from counting import LineCounter

VEHICLES = [2, 3, 5, 7]  # COCO class ids: car, motorcycle, bus, truck
WARMUP = 10  # first frames are slow (CUDA init, kernel selection): excluded from timing
SHORT_TRACK = 10  # tracks shorter than this (frames) are likely fragments or false positives
YELLOW = (0, 255, 255)  # BGR


def draw_counts(img, counter, line):
    cv2.line(img, *[tuple(p) for p in line], YELLOW, 2)
    for k, d in enumerate(counter.directions):
        by_cls = {c: n for (dd, c), n in sorted(counter.counts.items()) if dd == d}
        text = f"{d}: {sum(by_cls.values())}  " + " ".join(f"{c} {n}" for c, n in by_cls.items())
        org = (10, 25 + 25 * k)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # dark box behind the text (a thick black outline drifts: glyph spacing grows with thickness)
        cv2.rectangle(img, (org[0] - 4, org[1] - th - 5), (org[0] + tw + 4, org[1] + 6), (0, 0, 0), -1)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--scene", required=True, help="json with the counting line and direction names")
    p.add_argument("--out", default="runs/out.mp4")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    # low on purpose: ByteTrack's second pass needs weak detections; its own yaml thresholds do the filtering
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--tracker", default="configs/bytetrack.yaml")
    a = p.parse_args()

    scene = json.loads(Path(a.scene).read_text())
    counter = LineCounter(*scene["line"], directions=scene["directions"])

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    model = YOLO(a.model)

    frame_times, speeds, n_dets, crossings = [], [], [], []
    track_len = Counter()  # track id -> number of frames it was seen in
    frame_idx = 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()  # frame: numpy array (h, w, 3), uint8, BGR
        if not ok:
            break
        # persist=True: keep tracker state between calls, so ids carry over frame to frame.
        # agnostic_nms: one box per object across classes; per-class NMS kept "truck" + "bus" boxes
        # on the same lorry, and the tracker turned the extra box into a second id (double count).
        r = model.track(frame, persist=True, tracker=a.tracker, imgsz=a.imgsz, conf=a.conf,
                        classes=VEHICLES, agnostic_nms=True, device=0, verbose=False)[0]
        if r.boxes.id is not None:  # None when no box is matched to a track in this frame
            ids = r.boxes.id.int().tolist()
            track_len.update(ids)
            for (x1, y1, x2, y2), tid, c in zip(r.boxes.xyxy.tolist(), ids, r.boxes.cls.int().tolist()):
                # bottom-centre: where the vehicle touches the road, same plane as the painted line
                event = counter.update(tid, r.names[c], ((x1 + x2) / 2, y2))
                if event:
                    crossings.append((frame_idx, round(frame_idx / fps, 2), tid, event[1], event[0]))
        # thin boxes, short labels ("id:7 car"), one colour per track so each car is easy to follow
        img = r.plot(line_width=1, conf=False, color_mode="instance")
        draw_counts(img, counter, scene["line"])
        writer.write(img)
        frame_times.append(time.perf_counter() - t0)
        speeds.append(r.speed)  # ms: preprocess / inference / postprocess
        n_dets.append(len(r.boxes))
        frame_idx += 1

    cap.release()
    writer.release()

    csv_path = Path(a.out).with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "time_s", "track_id", "class", "direction"])
        out.writerows(crossings)

    t, s = frame_times[WARMUP:], speeds[WARMUP:]
    if not t:
        raise SystemExit("video too short to measure")
    print(f"frames: {len(frame_times)}  model: {a.model}  imgsz: {a.imgsz}  conf: {a.conf}")
    print(f"end-to-end: {len(t) / sum(t):.1f} FPS  ({1000 * sum(t) / len(t):.1f} ms/frame)")
    for k in ("preprocess", "inference", "postprocess"):
        print(f"  {k:<12} {sum(x[k] for x in s) / len(s):.1f} ms")
    print(f"vehicles per frame: mean {sum(n_dets) / len(n_dets):.1f}, max {max(n_dets)}")
    lens = sorted(track_len.values())
    if lens:
        short = sum(n < SHORT_TRACK for n in lens)
        print(f"tracks: {len(lens)} unique ids, {short} shorter than {SHORT_TRACK} frames, "
              f"median length {lens[len(lens) // 2]} frames")
    for d in counter.directions:
        by_cls = {c: n for (dd, c), n in sorted(counter.counts.items()) if dd == d}
        print(f"count {d}: {sum(by_cls.values())}  {by_cls}")
    print(f"saved: {a.out}, {csv_path}")


if __name__ == "__main__":
    main()
