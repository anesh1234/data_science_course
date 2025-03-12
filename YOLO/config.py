# YOLO hyperparameters:

# data (str): Path to dataset configuration file.
# epochs (int): Number of training epochs.
# batch_size (int): Batch size for training.# 
# imgsz (int): Input image size.
# device (str): Device to run training on (e.g., 'cuda', 'cpu').
# workers (int): Number of worker threads for data loading.
# optimizer (str): Optimizer to use for training.
# lr0 (float): Initial learning rate.
# patience (int): Epochs to wait for no observable improvement for early stopping of training.


DATA = "datasets/d1/data.yaml"
EPOCHS = 100
BATCH_SIZE = None
IMGSZ = None
DEVICE = None
WORKERS=None
OPTIMIZER=None
LR0=None
PATIENCE=None