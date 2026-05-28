# model-converter Docker Image

Docker image for converting YOLOv8 `.pt` models to ONNX and TensorRT FP16 engines.

---

## Requirements

- NVIDIA GPU with CUDA 12.x driver
- Docker installed
- NVIDIA Container Toolkit installed

### Install NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Build the Docker Image

Clone the repo and build from the Dockerfile:

```bash
git clone https://github.com/krishnakv24/pedestrian_analysis_cv_models.git
cd pedestrian_analysis_cv_models
docker build -t model-converter:latest .
```

Verify the image was built:

```bash
docker images model-converter
```

---

## What's Inside the Image

| Component | Version |
|---|---|
| Base Image | `nvcr.io/nvidia/tensorrt:24.08-py3` |
| TensorRT | 10.3 |
| CUDA | 12.6+ |
| Python | 3.10 |
| `ultralytics` | YOLOv8 PT → ONNX export |
| `onnx` + `onnxslim` | ONNX optimization |
| `onnxruntime` | ONNX validation |
| `trtexec` | ONNX → TensorRT build (built-in) |
| `libxcb1`, `libgl1`, `libglib2.0` | OpenCV headless system libs |

---

## Save Image as Tar (optional)

To export the built image for use on another machine:

```bash
docker save model-converter:latest -o model-converter.tar
```

To load it on another machine:

```bash
docker load -i model-converter.tar
```
