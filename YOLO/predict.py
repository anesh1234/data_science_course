from ultralytics import YOLO

# Load the YOLOv11 nano pretrained model
model = YOLO("YOLO/test/train4/weights/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
# Returns a list of Results objects
results = model(["datasets/d1/test/images/-1-_bmp.rf.0a1b63b0ef0ec1961342da2ff7908e12.jpg", 
                 "datasets/d1/test/images/-2-_jpg.rf.d0496bd87fc5593df878655de45dc9ac.jpg"])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk