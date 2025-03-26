from ultralytics import YOLO

# Load the YOLOv11 nano pretrained model
model = YOLO("YOLO/pt_nfrozen/train_id_3_part2/weights/best.pt")  # Our best model

# Run batched inference on a list of images
# Returns a list of Results objects
results = model(["datasets/final/valid/images/-63-_jpg.rf.87fd46fd7c7430910483d572be568b07.jpg", 
                 "datasets/final/valid/images/103_jpg.rf.b8b9120fb6b26e4743068d6d918c5c87.jpg",
                 "datasets/final/valid/images/0947_jpg.rf.ad5698df1fd26668334dfdf85afff5bf.jpg"])

# Process results list
for it, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename=f"Inference_result_{it}.jpg")  # save to disk