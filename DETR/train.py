from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()
import torch
import multiprocessing
import yaml

def main():

    # Possible configs:
    # 'npt_nfrozen' - Not pre-trained and no frozen layers
    # 'pt_frozen' - Pre-trained and frozen CNN backbone
    # 'pt_nfrozen' - Pre-trained and no frozen layers
    # config_name = 

    # Show GPU status
    DEVICE = None
    if torch.cuda.is_available():
        print('\nTorch can Access CUDA: TRUE\n')
        DEVICE = 'cuda'
    else:
        print('\nTorch can Access CUDA: FALSE\n')
        DEVICE = 'cpu'
    
    # Load a configuration YAML file
    # with open(f'DETR/{config_name}.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    # Load a YOLOv11 Nano model if it exists, downloads otherwise
    model = RTDETR()

    # Train the model
    # Documentation (comments on the right) were fetched from: 
    # https://docs.ultralytics.com/modes/train/#train-settings
    model.train(data= "datasets/final/data.yaml",  # data (str): Path to dataset configuration file.
                epochs= 300,                       # epochs (int): Number of training epochs. 
                batch= 30,                         # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
                imgsz= 640,                        # imgsz (int): Input image size.
                device= DEVICE,                    # device (str): Device to run training on (e.g., 'cuda', 'cpu').
                optimizer= 'Adam',                 # optimizer (str): Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration. Affects convergence speed and stability.
                lr0= 1,                            # lr0 (float): Initial learning rate.
                patience= 2,                       # patience (int): Epochs to wait for no observable improvement for early stopping of training.
                plots= True,                       # plots (bool): Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
                freeze= None,          # freeze (int or list): Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning.
                project= 'DETR/test'         # project (str): Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.
                )


# Idiom to ensure that a new process cannot start before the
# current process has finished its bootstrapping phase
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()