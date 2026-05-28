# model-converter Docker Image

Pre-built Docker image for converting YOLOv8 `.pt` models to ONNX and TensorRT FP16 engines.

---

## Image Details

| Property | Value |
|---|---|
| File | `model-converter-latest.tar` |
| Size | 8.1 GB |
| Base Image | `nvcr.io/nvidia/tensorrt:24.08-py3` |
| TensorRT Version | 10.3 |
| CUDA | 12.6+ |
| Python | 3.10 |

### Included Packages

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 PT → ONNX export |
| `onnx` | ONNX model loading and inspection |
| `onnxslim` | ONNX graph simplification |
| `onnxruntime` | ONNX validation |
| `trtexec` | ONNX → TensorRT engine build (built-in) |
| `libxcb1`, `libgl1`, `libglib2.0` | OpenCV headless system libs |

---

## Load the Image

```bash
docker load -i model-converter-latest.tar
```

Verify it loaded:

```bash
docker images model-converter
```

---

## Converting Models

> **Important:** Run ONNX export and TensorRT build as **two separate `docker run` commands**.
> PyTorch holds GPU/CPU memory during export — running TRT in the same process can cause failures.

### Stage 1 — PT → ONNX

```bash
docker run --rm --gpus all \
  -v /path/to/ParentModel:/models/ParentModel \
  -v /path/to/OnnxModels:/models/OnnxModels \
  -v /path/to/TensorRTModels:/models/TensorRTModels \
  -v /path/to/convert.py:/workspace/convert.py \
  model-converter:latest \
  python convert.py --onnx-only
```

### Stage 2 — ONNX → TensorRT Engine

```bash
docker run --rm --gpus all \
  -v /path/to/OnnxModels:/models/OnnxModels \
  -v /path/to/TensorRTModels:/models/TensorRTModels \
  -v /path/to/convert.py:/workspace/convert.py \
  model-converter:latest \
  python convert.py --trt-only
```

### Single Model

```bash
docker run --rm --gpus all \
  -v /path/to/ParentModel:/models/ParentModel \
  -v /path/to/OnnxModels:/models/OnnxModels \
  -v /path/to/TensorRTModels:/models/TensorRTModels \
  -v /path/to/convert.py:/workspace/convert.py \
  model-converter:latest \
  python convert.py --model yolov8n.pt
```

---

## TensorRT Engine Settings

| Setting | Value |
|---|---|
| Precision | FP16 |
| Min batch | 1 |
| Opt batch | 4 |
| Max batch | 4 |
| Input shape | N × 3 × 640 × 640 |
| Workspace | 4096 MiB |

---

## Requirements

- NVIDIA GPU (tested: RTX 4050 6GB)
- CUDA driver 12.x
- Docker with NVIDIA Container Toolkit installed
- WSL2 or Linux host

### Install NVIDIA Container Toolkit (if not already installed)

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
