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
    # Comments with stars indicate that the parameter values were calculated using the "tune" method
    model.train(data= "datasets/final/data.yaml",  # data (str): Path to dataset configuration file.
                epochs= 300,                       # epochs (int): Number of training epochs. 
                batch= 40,                         # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
                imgsz= 640,                        # imgsz (int): Input image size.
                device= DEVICE,                    # device (str): Device to run training on (e.g., 'cuda', 'cpu').
                optimizer= 'Adam',                 # optimizer (str): Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration. Affects convergence speed and stability.    
                patience= 20,                      # patience (int): Epochs to wait for no observable improvement for early stopping of training.
                plots= True,                       # plots (bool): Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
                freeze= config['freeze'],          # freeze (int or list): Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning.
                project= config['project'],        # project (str): Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.
                lr0= 0.01056,                      # * Initial learning rate (i.e. SGD=1E-2, Adam=1E-3). Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
                lrf= 0.00705,                      # * Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.
                momentum= 0.81053,                 # * Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update.
                weight_decay= 0.00054,             # * L2 regularization term, penalizing large weights to prevent overfitting.
                warmup_epochs= 2.62476,            # * Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.
                warmup_momentum= 0.70931,          # * Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.
                box= 6.88736,                      # * Weight of the box loss component in the loss function, influencing how much emphasis is placed on accurately predicting bounding box coordinates.
                cls= 0.4794,                       # * Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.
                dfl= 0.72781,                      # * Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.
                hsv_h= 0.01524,                    # * Augmentation: Adjusts the hue of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.
                hsv_s= 0.56541,                    # * Augmentation: Alters the saturation of the image by a fraction, affecting the intensity of colors. Useful for simulating different environmental conditions.
                hsv_v= 0.48063,                    # * Augmentation: Modifies the value (brightness) of the image by a fraction, helping the model to perform well under various lighting conditions.
                translate= 0.07406,                # * Augmentation: Translates the image horizontally and vertically by a fraction of the image size, aiding in learning to detect partially visible objects.
                scale= 0.28985,                    # * Augmentation: Scales the image by a gain factor, simulating objects at different distances from the camera.
                fliplr= 0.47655,                   # * Augmentation: Flips the image left to right with the specified probability, useful for learning symmetrical objects and increasing dataset diversity.
                mosaic= 1                          # * Augmentation: Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.
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
    main(yolo_train_batch[1])