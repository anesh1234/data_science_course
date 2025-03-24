from ultralytics import YOLO
import torch
import multiprocessing
import yaml

def main(config_name:str):

    # Show GPU status
    DEVICE = None
    if torch.cuda.is_available():
        print('\nCUDA is Available: TRUE\n')
        DEVICE = 'cuda'
    else:
        print('\nCUDA is Available: FALSE\n')
        DEVICE = 'cpu'
    
    # Load a configuration YAML file
    with open(f'YOLO/{config_name}.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load a YOLOv11 Nano model if it exists, downloads otherwise
    model = YOLO(model=config['model'], task='detect')

    # Train the model
    # Documentation (comments on the right) were fetched from: 
    # https://docs.ultralytics.com/modes/train/#train-settings
    model.train(data= "datasets/final/data.yaml",  # data (str): Path to dataset configuration file.
                epochs= 300,                       # epochs (int): Number of training epochs. 
                batch= 60,                         # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
                imgsz= 640,                        # imgsz (int): Input image size.
                device= DEVICE,                    # device (str): Device to run training on (e.g., 'cuda', 'cpu').
                optimizer= 'Adam',                 # optimizer (str): Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration. Affects convergence speed and stability.    
                patience= 20,                      # patience (int): Epochs to wait for no observable improvement for early stopping of training.
                plots= True,                       # plots (bool): Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
                freeze= config['freeze'],          # freeze (int or list): Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning.
                project= config['project'],        # project (str): Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.
                lr0= 0.00956,
                lrf= 0.01013,
                momentum= 0.97294,
                weight_decay= 0.00043,
                warmup_epochs= 3.08681,
                warmup_momentum= 0.82874,
                box= 7.75731,
                cls= 0.4832,
                dfl= 1.46943,
                hsv_h= 0.01507,
                hsv_s= 0.65767,
                hsv_v= 0.42403,
                degrees= 0.0,
                translate= 0.09804,
                scale= 0.52976,
                shear= 0.0,
                perspective= 0.0,
                flipud= 0.0,
                fliplr= 0.58352,
                bgr= 0.0,
                mosaic= 0.99633,
                mixup= 0.0,
                copy_paste= 0.0
                )



# Idiom to ensure that a new process cannot start before the
# current process has finished its bootstrapping phase
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Possible configs:
    # 'npt_nfrozen' - Not pre-trained and no frozen layers
    # 'pt_frozen' - Pre-trained and frozen CNN backbone
    # 'pt_nfrozen' - Pre-trained and no frozen layers

    yolo_train_batch = ['npt_nfrozen', 'pt_nfrozen', 'pt_frozen']
    main(yolo_train_batch[2])