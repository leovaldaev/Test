from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO('runs/detect/yolo11s.pt_100_1280_b_0.8_14nov_1/weights/best.pt')

# Export the model to TensorRT
# model.export(format="engine", imgsz=1280)  # creates 'yolo11n.engine'

