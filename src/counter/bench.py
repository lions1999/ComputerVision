"""Model-only latency of each export: same 1x3x640x640 input, same Python path (ultralytics AutoBackend), batch 1.

Measures the network forward pass only: no video decoding, letterbox, NMS or tracking (see run.py for end-to-end).
"""
import argparse
import statistics
import time

import torch
from ultralytics.nn.autobackend import AutoBackend

VARIANTS = [  # (label, weights, device, fp16)
    ("PyTorch FP32", "models/yolo11n.pt", "cuda:0", False),
    ("PyTorch FP16", "models/yolo11n.pt", "cuda:0", True),
    ("ONNX Runtime GPU FP32", "models/yolo11n.onnx", "cuda:0", False),
    ("TensorRT FP16", "models/yolo11n_fp16.engine", "cuda:0", False),  # precision is baked into the engine
    ("TensorRT INT8", "models/yolo11n_int8.engine", "cuda:0", False),
    ("ONNX Runtime CPU FP32", "models/yolo11n.onnx", "cpu", False),
]


def bench(weights, device, fp16, warmup, iters):
    dev = torch.device(device)
    model = AutoBackend(weights, device=dev, fp16=fp16, verbose=False)
    # content does not change convolution timings; a fixed square input keeps every backend on the same shape
    # (YOLO('x.pt').predict would letterbox to 640x384 for PyTorch only, flattering it)
    im = torch.rand(1, 3, 640, 640, device=dev)
    im = im.half() if fp16 else im
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    times = []
    for k in range(warmup + iters):
        sync()  # GPU calls are asynchronous: without a sync the timer would only measure the kernel launch
        t0 = time.perf_counter()
        model(im)
        sync()
        if k >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return statistics.median(times), times[int(0.95 * len(times)) - 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=500)
    a = p.parse_args()

    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, {a.iters} runs after {a.warmup} warmup")
    print(f"{'variant':<24}{'median ms':>10}{'p95 ms':>9}{'FPS':>8}{'speedup':>9}")
    base = None
    for label, weights, device, fp16 in VARIANTS:
        med, p95 = bench(weights, device, fp16, a.warmup, a.iters if device != "cpu" else a.iters // 5)
        base = base or med
        print(f"{label:<24}{med:>10.2f}{p95:>9.2f}{1000 / med:>8.0f}{base / med:>8.1f}x")


if __name__ == "__main__":
    main()
