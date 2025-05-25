import os, sys, yaml
from ultralytics import YOLO

PT_PATH   = "tes1.pt"
YAML_PATH = "data.yaml"
ENGINE_OUT = "tes1.engine"
IMG_SIZE  = 640
USE_FP16  = True

try:
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names, nc = y["names"], y["nc"]
    assert len(names) == nc
except Exception as e:
    sys.exit(f"YAML error: {e}")

model = YOLO(PT_PATH)
model.model.names = names
model.model.nc = nc
print(f"loaded {PT_PATH} | nc={nc} classes")

engine_path = model.export(
    format   = "engine",
    imgsz    = IMG_SIZE,
    half     = USE_FP16,
    device   = "0",
    simplify = True,
    dynamic  = False,
)

if engine_path != ENGINE_OUT:
    os.rename(engine_path, ENGINE_OUT)
print(f"export complete -> {ENGINE_OUT}")
