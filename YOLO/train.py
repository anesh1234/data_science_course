from ultralytics import YOLO
import torch
import multiprocessing
from pathlib import Path
import os

def main():
    # Show GPU status
    DEVICE = None
    if torch.cuda.is_available():
        print('\nTorch can Access CUDA: TRUE\n')
        DEVICE = 'cuda'
    else:
        print('\nTorch can Access CUDA: FALSE\n')
        DEVICE = 'cpu'

    # Load the YOLOv11 nano pretrained model if it exists,
    # downloads otherwise
    model = YOLO("yolo11n.pt")

    # Train the model
    results = model.train(data= "datasets/d1/data.yaml",  # data (str): Path to dataset configuration file.
                        epochs= 5,                       # epochs (int): Number of training epochs., 
                        batch= 10,                        # batch (int): Batch size for training.
                        imgsz= 640,                       # imgsz (int): Input image size.
                        device= DEVICE,                   # device (str): Device to run training on (e.g., 'cuda', 'cpu').
                        optimizer= 'auto',                # optimizer (str): Optimizer to use for training.
                        lr0= None,                        # lr0 (float): Initial learning rate.
                        patience= None,                   # patience (int): Epochs to wait for no observable improvement for early stopping of training.
                        plots= True                       # plots (bool): Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
                        )

    # YOLO hyperparameters:
    # https://docs.ultralytics.com/modes/train/#train-settings


# Idiom to ensure that a new process cannot start before the
# current process has finished its bootstrapping phase
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()