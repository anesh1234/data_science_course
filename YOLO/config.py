# YOLO hyperparameters:
# https://docs.ultralytics.com/modes/train/#train-settings


#task=detect, mode=train, time=None, save=True, save_period=-1, cache=False, device=None, workers=None, project=None, name=train11, exist_ok=False, pretrained=True, optimizer=None, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, lr0=None, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml,

DATA = "datasets/d1/data.yaml"  # data (str): Path to dataset configuration file.
EPOCHS = 5  # epochs (int): Number of training epochs.
BATCH_SIZE = 10  # batch (int): Batch size for training.
IMGSZ = 640  # imgsz (int): Input image size.
DEVICE = None  # device (str): Device to run training on (e.g., 'cuda', 'cpu').
WORKERS=None  # workers (int): Number of worker threads for data loading.
OPTIMIZER='auto'  # optimizer (str): Optimizer to use for training.
LR0=None  # lr0 (float): Initial learning rate.
PATIENCE=None  # patience (int): Epochs to wait for no observable improvement for early stopping of training.
SAVE_DIR='YOLO\detect\train11'