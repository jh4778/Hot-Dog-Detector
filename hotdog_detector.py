from ultralytics import YOLO

# 2. Load a pretrained YOLOv8 model (tiny version for speed)
model = YOLO("yolov8n.pt")

# 3. Train the model on your dataset
# Make sure you have hotdog.yaml file configured (see below)
model.train(
    data="hotdog.yaml",   # path to dataset config
    epochs=20,            # increase if you want more accuracy
    imgsz=640,            # image size (standard for YOLO)
    batch=16              # adjust if your GPU/CPU has limited memory
)

# 4. Validate the model on validation set
metrics = model.val()
print(metrics)  # includes mAP (mean average precision)

# 5. Run inference on a test image
results = model("test_hotdog.jpg")  # replace with your test image path
results[0].show()                      # shows the image with boxes
results[0].save("outputs/")            # saves the annotated image(s)
