from ultralytics import YOLO

# load model
model = YOLO("yolov8n.pt")

# export to ONNX
model.export(format="onnx", opset=12)