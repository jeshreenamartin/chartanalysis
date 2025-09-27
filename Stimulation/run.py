import os
from classification import classify_retina
from seg import segment_retina
from detection import detect_retina


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

img_path = os.path.join(BASE_DIR, "datasets", "sample_retina.png")  # or .jpg

if not os.path.exists(img_path):
    print(f"❌ Image not found at {img_path}")
    exit(1)
else:
    print(f"✅ Found image at {img_path}")


classification_result = classify_retina(img_path)
print("\n🎯 Classification Result:")
print(classification_result)

segmentation_mask = segment_retina(img_path)
print("\n🖌 Segmentation Mask Shape:", segmentation_mask.shape)

detection_result = detect_retina(img_path)
print("\n🔍 Detection Result:")
print(detection_result)
