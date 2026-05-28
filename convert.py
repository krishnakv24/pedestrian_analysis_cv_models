import shutil
import subprocess
from pathlib import Path

PARENT_DIR  = Path("/models/ParentModel")
ONNX_DIR    = Path("/models/OnnxModels")
TRT_DIR     = Path("/models/TensorRTModels")

ONNX_DIR.mkdir(parents=True, exist_ok=True)
TRT_DIR.mkdir(parents=True, exist_ok=True)

def pt_to_onnx(pt_path: Path) -> Path:
    from ultralytics import YOLO
    onnx_out = ONNX_DIR / f"{pt_path.stem}.onnx"
    if onnx_out.exists():
        print(f"[SKIP] {onnx_out.name} already exists")
        return onnx_out

    print(f"[ONNX] Converting {pt_path.name} ...")
    model = YOLO(str(pt_path))
    model.export(format="onnx", dynamic=True, simplify=True)

    # ultralytics writes the .onnx next to the .pt file
    exported = pt_path.parent / f"{pt_path.stem}.onnx"
    if exported.exists():
        shutil.move(str(exported), str(onnx_out))
    print(f"[ONNX] Saved -> {onnx_out}")
    return onnx_out


def get_onnx_input_name(onnx_path: Path) -> str:
    import onnx
    model = onnx.load(str(onnx_path))
    return model.graph.input[0].name


def onnx_to_trt(onnx_path: Path) -> Path:
    trt_out = TRT_DIR / f"{onnx_path.stem}.engine"
    if trt_out.exists():
        print(f"[SKIP] {trt_out.name} already exists")
        return trt_out

    input_name = get_onnx_input_name(onnx_path)
    print(f"[TRT]  Converting {onnx_path.name} (input: {input_name}) ...")
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_out}",
        "--fp16",
        "--memPoolSize=workspace:4096",
        f"--minShapes={input_name}:1x3x640x640",
        f"--optShapes={input_name}:4x3x640x640",
        f"--maxShapes={input_name}:4x3x640x640",
    ]
    subprocess.run(cmd, check=True)
    print(f"[TRT]  Saved -> {trt_out}")
    return trt_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Single model filename e.g. yolov8n.pt")
    args = parser.parse_args()

    if args.model:
        pt_files = [PARENT_DIR / args.model]
        if not pt_files[0].exists():
            print(f"Model not found: {pt_files[0]}")
            raise SystemExit(1)
    else:
        pt_files = sorted(PARENT_DIR.glob("*.pt"))
        if not pt_files:
            print("No .pt files found in ParentModel/")
            raise SystemExit(1)

    for pt_file in pt_files:
        print(f"\n{'='*50}")
        print(f"Model: {pt_file.name}")
        onnx_path = pt_to_onnx(pt_file)
        onnx_to_trt(onnx_path)

    print(f"\n{'='*50}")
    print("All conversions complete.")
    print(f"  ONNX    -> {ONNX_DIR}")
    print(f"  TensorRT-> {TRT_DIR}")
