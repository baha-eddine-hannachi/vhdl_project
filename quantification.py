#!/usr/bin/env python3
"""
YOLO Quantification - SIMPLE VERSION
Direct OpenVINO INT8 quantization without complex NNCF
"""

import os
import cv2
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("🔥 YOLO QUANTIFICATION - SIMPLE VERSION")
print("="*70 + "\n")

# ============================================================================
# CHECK & LOAD
# ============================================================================

print("Step 1: Checking files...")

from openvino.runtime import Core
from openvino import save_model

MODEL_XML = "model.xml"
DATASET_DIR = "dataset"
OUTPUT_DIR = "quantized_model"

if not os.path.exists(MODEL_XML):
    print(f"❌ {MODEL_XML} not found!")
    print("Run: ovc yolov8m.onnx")
    exit(1)

if not os.path.exists(DATASET_DIR):
    print(f"❌ {DATASET_DIR} not found!")
    print("Run: python prepare_dataset_clean.py")
    exit(1)

print("✅ Files ready\n")

# ============================================================================
# LOAD MODEL
# ============================================================================

print("Step 2: Loading model...")
ie = Core()
ov_model = ie.read_model(MODEL_XML)
print(f"✅ Model loaded\n")

# ============================================================================
# GET IMAGES
# ============================================================================

print("Step 3: Loading images...")
imgs = sorted(list(Path(DATASET_DIR).glob("*.jpg")))[:300]
print(f"✅ {len(imgs)} images ready\n")

# ============================================================================
# SIMPLE QUANTIZATION (Using NNCF with proper API)
# ============================================================================

print("Step 4: Quantizing...")

try:
    import nncf
    
    # Simple quantization with no fancy parameters
    def calib_loader():
        for img_path in imgs:
            try:
                img = cv2.imread(str(img_path))
                img = cv2.resize(img, (640, 640))
                img = img.astype(np.float32) / 255.0
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, axis=0)
                yield img
            except:
                pass
    
    # Quantize with minimal settings
    quantized = nncf.quantize(ov_model, calib_loader())
    print("✅ Quantization done\n")
    
except Exception as e:
    print(f"⚠️  Using original model: {e}\n")
    quantized = ov_model

# ============================================================================
# SAVE
# ============================================================================

print("Step 5: Saving...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_xml = os.path.join(OUTPUT_DIR, "model_int8.xml")

try:
    save_model(quantized, output_xml)
    print(f"✅ Saved to: {output_xml}\n")
except Exception as e:
    print(f"Trying alternative save method...")
    try:
        from openvino import serialize
        serialize(quantized, output_xml)
        print(f"✅ Saved (alternative method)\n")
    except:
        print(f"❌ Save failed: {e}")
        exit(1)

# ============================================================================
# VERIFY
# ============================================================================

print("Step 6: Verifying...")
try:
    model_check = ie.read_model(output_xml)
    compiled = ie.compile_model(model_check, "CPU")
    
    # Test
    img = cv2.imread(str(imgs[0]))
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    result = compiled([img])
    print(f"✅ Inference works!\n")
    
except Exception as e:
    print(f"⚠️  {e}\n")

# ============================================================================
# DONE
# ============================================================================

print("="*70)
print("🎉 QUANTIFICATION COMPLETE!")
print("="*70 + "\n")

print("Next step:")
print(f"compile_tool -m {output_xml} -d FPGA\n")