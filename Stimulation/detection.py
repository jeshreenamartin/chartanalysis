from ultralytics import YOLO


model = YOLO("yolov8n.pt")  
def detect_retina(img_path):
    results = model(img_path)
    boxes = []
    for box in results[0].boxes:
        boxes.append({
            "bbox": box.xyxy.tolist(),
            "confidence": float(box.conf),
            "class_id": int(box.cls)
        })
    return boxes


print(detect_retina("./datasets/sample_retina.png"))
