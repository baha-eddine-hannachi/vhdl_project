from openvino.runtime import Core
import cv2
import numpy as np

# 1. Load Model
ie = Core()
# git add README.mdEnsure your path uses / or double \\ to avoid string errors
model = ie.read_model(r"quantized_model\model_int8.xml") 
compiled_model = ie.compile_model(model, "CPU")
output_layer = compiled_model.output(0)

# 2. Setup Webcam
cap = cv2.VideoCapture(0)

def preprocess(frame):
    # Resize to model input size (640x640 for YOLOv8/v11)
    img = cv2.resize(frame, (640, 640))
    # Change format from HWC to CHW and add batch dimension
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    # Convert to float32 (OpenVINO expects this even for INT8 models)
    img = img.astype(np.float32) / 255.0
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    input_data = preprocess(frame)

    # 3. Inference
    results = compiled_model([input_data])[output_layer]
    
    # YOLO output is usually [1, 84, 8400] or [1, detections, 6]
    # We squeeze to remove the batch dimension
    outputs = np.squeeze(results) 

    # Note: Modern YOLO (v8+) outputs are transposed (84 x 8400)
    # If your boxes look crazy, you might need: outputs = outputs.T
    
    # 4. Detection & Drawing (Inside the Loop!)
    # This is a simplified loop for demonstration
    for det in outputs.T: 
        # Standard YOLOv8 format: [x, y, w, h, class_scores...]
        conf = np.max(det[4:]) 
        if conf > 0.5:
            # Rescale coordinates to original frame size
            # YOLO coordinates are usually center_x, center_y, width, height (0 to 1)
            cx, cy, bw, bh = det[:4]
            
            x1 = int((cx - bw/2) * w / 640)
            y1 = int((cy - bh/2) * h / 640)
            x2 = int((cx + bw/2) * w / 640)
            y2 = int((cy + bh/2) * h / 640)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. THE FIX: Show the frame
    cv2.imshow("OpenVINO YOLO Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()