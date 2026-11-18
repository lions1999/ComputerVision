"""Manual ground truth: step through the original video and log each vehicle crossing the counting line."""
import argparse
import csv
import json
from collections import deque
from pathlib import Path

import cv2

KEYS = {"c": "car", "t": "truck", "b": "bus", "m": "motorcycle"}  # same names as YOLO's COCO classes
# shortcut: only the last BACK frames can be revisited (all frames of a 1080p video won't fit in RAM)
BACK = 150
YELLOW = (0, 255, 255)
WIN = "annotate"


def put(img, text, y):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (6, y - th - 5), (14 + tw, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--scene", required=True, help="json with the counting line and direction names")
    p.add_argument("--out", required=True, help="csv; if it exists, annotation resumes from it")
    p.add_argument("--start", type=float, default=0, help="seconds")
    p.add_argument("--end", type=float, default=None, help="seconds")
    a = p.parse_args()

    scene = json.loads(Path(a.scene).read_text())
    line, dirs = scene["line"], scene["directions"]
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    end_frame = round(a.end * fps) if a.end is not None else None

    out = Path(a.out)
    rows = []  # [frame, time_s, class, direction]
    if out.exists():  # never throw away manual work: resume
        with open(out, newline="") as f:
            rows = [[int(r["frame"]), float(r["time_s"]), r["class"], r["direction"]] for r in csv.DictReader(f)]
        print(f"resuming: {len(rows)} crossings loaded from {out}")

    # read-and-discard rather than seeking: seeking in compressed video can land on the wrong frame
    for _ in range(round(a.start * fps)):
        cap.grab()
    next_idx = round(a.start * fps)  # frame indices are absolute, like run.py's CSV
    buf, pos = deque(maxlen=BACK), -1  # recent (frame_idx, image); pos indexes into buf

    def forward():
        nonlocal pos, next_idx
        if pos < len(buf) - 1:  # we had stepped back: replay from the buffer
            pos += 1
            return True
        if end_frame is not None and next_idx >= end_frame:
            return False
        ok, frame = cap.read()
        if not ok:
            return False
        buf.append((next_idx, frame))
        next_idx += 1
        pos = len(buf) - 1
        return True

    print(f"space play/pause | d / a: next / previous frame | u: undo | q: save and quit\n"
          f"c t b m = car truck bus motorcycle -> {dirs[0]};  Shift+key -> {dirs[1]}")
    if not forward():
        raise SystemExit("no frames to show")
    paused = True
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)  # resizable: a 1080p frame may not fit the screen
    while True:
        fidx, frame = buf[pos]
        img = frame.copy()
        cv2.line(img, *[tuple(pt) for pt in line], YELLOW, 1)
        put(img, f"frame {fidx}  t {fidx / fps:.2f}s  {'PAUSED' if paused else ''}", 25)
        for k, d in enumerate(dirs):
            put(img, f"{d}: {sum(r[3] == d for r in rows)}", 50 + 25 * k)
        for k, r in enumerate(rows[-3:]):
            put(img, f"last: frame {r[0]} {r[2]} {r[3]}", 110 + 25 * k)
        cv2.imshow(WIN, img)

        key = cv2.waitKey(0 if paused else max(1, round(1000 / fps)))
        ch = chr(key & 0xFF) if key != -1 else ""
        if key == -1:  # timeout while playing
            if not forward():
                paused = True
        elif ch == " ":
            paused = not paused
        elif ch == "d":
            paused = True
            forward()
        elif ch == "a":
            paused = True
            pos = max(pos - 1, 0)
        elif ch == "u" and rows:
            print("undo:", rows.pop())
        elif ch in ("q", "\x1b") or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif ch.lower() in KEYS:
            row = [fidx, round(fidx / fps, 2), KEYS[ch.lower()], dirs[0] if ch.islower() else dirs[1]]
            rows.append(row)
            print("added:", row)

    cap.release()
    cv2.destroyAllWindows()
    rows.sort()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "class", "direction"])
        w.writerows(rows)
    for d in dirs:
        print(f"{d}: {sum(r[3] == d for r in rows)}")
    print(f"saved: {out} ({len(rows)} crossings)")


if __name__ == "__main__":
    main()
