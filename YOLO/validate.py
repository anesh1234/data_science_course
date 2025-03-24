from ultralytics import YOLO
import multiprocessing

def main(train_id, config):
    
    # Load a model
    model = YOLO("YOLO/pt_frozen/train_id_2/weights/best.pt")  # load a custom model

    metrics = model.val(project = f'YOLO/{config}/val_id_{train_id}')



# Idiom to ensure that a new process cannot start before the
# current process has finished its bootstrapping phase
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Possible configs:
    # 'npt_nfrozen' - Not pre-trained and no frozen layers
    # 'pt_frozen' - Pre-trained and frozen CNN backbone
    # 'pt_nfrozen' - Pre-trained and no frozen layers

    train_id = 
    config = 
    main(train_id, config)