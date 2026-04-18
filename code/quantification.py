from openvino.tools import mo
from openvino.runtime import Core

onnx_model = "yolov8n.onnx"
output_dir = "./ir_model"

print("🔄 Converting ONNX → OpenVINO IR...")
mo.convert_model(
    onnx_model,
    output_dir=output_dir,
    model_name="yolov8n",
    data_type="INT8"  
)

print(" Conversion successful!")
print(f"Files created: {output_dir}/yolov8n.xml و {output_dir}/yolov8n.bin")