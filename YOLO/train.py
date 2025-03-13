from test.test_itertools import batched
from ultralytics import YOLO
from config import *

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=DATA, 
                      epochs=EPOCHS, 
                      batch=BATCH_SIZE,
                      imgsz=IMGSZ,
                      device=DEVICE,
                      workers=WORKERS,
                      optimizer=OPTIMIZER,
                      lr0=LR0,
                      patience=PATIENCE,
                      save_dir=SAVE_DIR
                      )