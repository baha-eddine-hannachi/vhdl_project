#!/usr/bin/env python3
"""
Dataset Preparation Script
Extract 300 face images from CelebA for quantization calibration
"""

import os
import shutil
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

SOURCE_FOLDER = "img_align_celeba"  # CelebA source
OUTPUT_FOLDER = "dataset"            # Output for quantization
NUM_IMAGES = 300                     # How many to extract

# ============================================================================
# STARTUP
# ============================================================================

print("\n" + "="*70)
print("📁 DATASET PREPARATION - CelebA")
print("="*70 + "\n")

# ============================================================================
# STEP 1: CHECK SOURCE
# ============================================================================

print("🔍 Step 1: Checking CelebA source...")

if not os.path.exists(SOURCE_FOLDER):
    print(f"❌ Source folder '{SOURCE_FOLDER}' not found!")
    print("\nCreate folder with CelebA images:")
    print("  img_align_celeba/")
    print("    ├─ 000001.jpg")
    print("    ├─ 000002.jpg")
    print("    └─ ... (many more)")
    exit(1)

# Get all images
all_images = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
    all_images.extend(Path(SOURCE_FOLDER).glob(ext))

all_images = sorted(list(all_images))

print(f"✅ Found {len(all_images)} images in CelebA\n")

if len(all_images) == 0:
    print("❌ No images found!")
    exit(1)

# ============================================================================
# STEP 2: CREATE OUTPUT FOLDER
# ============================================================================

print("📂 Step 2: Creating output folder...")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"✅ Created: {OUTPUT_FOLDER}\n")

# ============================================================================
# STEP 3: COPY IMAGES
# ============================================================================

print(f"📋 Step 3: Copying {NUM_IMAGES} images...")

# Select images
selected = all_images[:min(NUM_IMAGES, len(all_images))]

# Copy
copied = 0
for idx, img_path in enumerate(selected):
    dest = os.path.join(OUTPUT_FOLDER, img_path.name)
    
    try:
        shutil.copy(str(img_path), dest)
        copied += 1
        
        # Progress
        if (idx + 1) % 50 == 0 or (idx + 1) == len(selected):
            print(f"   {idx + 1}/{len(selected)} images...")
    
    except Exception as e:
        print(f"   ⚠️  Skipped {img_path.name}: {e}")

print(f"\n✅ Copied {copied} images to '{OUTPUT_FOLDER}'\n")

# ============================================================================
# VERIFICATION
# ============================================================================

print("✅ Step 4: Verifying...")

output_images = list(Path(OUTPUT_FOLDER).glob("*.jpg"))
output_images.extend(Path(OUTPUT_FOLDER).glob("*.png"))

print(f"✅ Total images in {OUTPUT_FOLDER}: {len(output_images)}\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("="*70)
print("✅ DATASET PREPARATION COMPLETE!")
print("="*70 + "\n")

print(f"📊 Summary:")
print(f"   ✓ Source: {SOURCE_FOLDER}")
print(f"   ✓ Output: {OUTPUT_FOLDER}")
print(f"   ✓ Images: {len(output_images)}\n")

print("⏭️  Next Step:")
print("   python quantize.py\n")

print("="*70)