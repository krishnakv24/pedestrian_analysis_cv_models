# Model Repo

AI inference model repository for person detection, face detection, and face re-identification.
Supports conversion from PyTorch `.pt` → ONNX → TensorRT FP16 engines via Docker.

---

## Folder Structure

```
Model_Repo/
├── ParentModel/          # Source PyTorch models (.pt)
├── OnnxModels/           # Exported ONNX models (.onnx)
├── TensorRTModels/       # Compiled TensorRT engines (.engine)
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Compose config with GPU support
├── convert.py            # Conversion script (PT → ONNX → TRT)
├── model-converter.tar   # Pre-built Docker image
└── README.md
```

---

## Models

| Model File | Purpose | Input Shape |
|---|---|---|
| `yolov8n.pt` | Person detection | 640×640 |
| `yolov8l-face-lindevs.pt` | Face detection | 640×640 |
| `yolov8n-face.pt` | Face Re-ID | 640×640 |

---

## Docker Setup

### Load the pre-built image

```bash
docker load -i model-converter.tar
```

### Or build from source

```bash
docker build -t model-converter:latest .
```

---

## Running Conversions

Conversions are split into two stages to avoid memory conflicts between PyTorch and TensorRT.

### Stage 1 — Export PT → ONNX

```bash
docker run --rm --gpus all \
  -v ./ParentModel:/models/ParentModel \
  -v ./OnnxModels:/models/OnnxModels \
  -v ./TensorRTModels:/models/TensorRTModels \
  -v ./convert.py:/workspace/convert.py \
  model-converter:latest \
  python convert.py --onnx-only
```

### Stage 2 — Build ONNX → TensorRT Engine

```bash
docker run --rm --gpus all \
  -v ./OnnxModels:/models/OnnxModels \
  -v ./TensorRTModels:/models/TensorRTModels \
  -v ./convert.py:/workspace/convert.py \
  model-converter:latest \
  python convert.py --trt-only
```

### Convert a single model

```bash
docker run --rm --gpus all \
  -v ./ParentModel:/models/ParentModel \
  -v ./OnnxModels:/models/OnnxModels \
  -v ./TensorRTModels:/models/TensorRTModels \
  -v ./convert.py:/workspace/convert.py \
  model-converter:latest \
  python convert.py --model yolov8n.pt
```

---

## TensorRT Engine Settings

| Setting | Value |
|---|---|
| Precision | FP16 |
| Min batch size | 1 |
| Opt batch size | 4 |
| Max batch size | 4 |
| Input shape | 1–4 × 3 × 640 × 640 |
| Workspace | 4096 MiB |

---

## Output Models

| ONNX | Engine | Size (ONNX / Engine) |
|---|---|---|
| `yolov8n.onnx` | `yolov8n.engine` | 13M / 9.6M |
| `yolov8l-face-lindevs.onnx` | `yolov8l-face-lindevs.engine` | 167M / 87M |
| `yolov8n-face.onnx` | `yolov8n-face.engine` | 12M / 9.2M |

---

## Requirements

- NVIDIA GPU (tested on RTX 4050)
- CUDA 12.x driver
- Docker with NVIDIA Container Toolkit
- TensorRT 10.3 (included in Docker image)
