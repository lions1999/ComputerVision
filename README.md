# Vehicle Counter

Counts vehicles by class in traffic video: YOLO detection, multi-object tracking,
line-crossing count. Accuracy is measured against a manual count, then the model
is optimized for edge-style inference (ONNX, TensorRT FP16/INT8) and the
speed/accuracy trade-off is measured.

**Status:** stage 0 (environment) done.

## Setup

Tested on Windows 11, RTX 3050 Laptop GPU (4 GB, 30 W), NVIDIA driver 596.49,
Python 3.12, PyTorch 2.11 + CUDA 12.8.

No CUDA Toolkit install is needed: the PyTorch wheels ship the CUDA runtime.
The driver only has to support CUDA ≥ 12.8 (check the top-right of `nvidia-smi`).

```powershell
uv python install 3.12
uv venv --python 3.12 --seed .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` points pip at PyTorch's index for the CUDA build of torch;
installing `ultralytics` alone from PyPI would pull a CPU-only torch.

Check the GPU is visible:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
yolo predict model=yolo11n.pt source=https://ultralytics.com/images/bus.jpg device=0
```

Expected: `True NVIDIA GeForce RTX 3050 ...`, then `CUDA:0` in the YOLO log and
annotated output in `runs/detect/predict/`.

## License note

Ultralytics YOLO is AGPL-3.0; this repository is published as open source accordingly.
