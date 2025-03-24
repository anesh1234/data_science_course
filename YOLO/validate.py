from ultralytics import YOLO
import multiprocessing

def main():
    
    # Load a model
    model = YOLO("YOLO/pt_frozen/train_id_2/weights/best.pt")  # load a custom model

    metrics = model.val(project = 'YOLO/validationTest',
                        save_json = True)



# Idiom to ensure that a new process cannot start before the
# current process has finished its bootstrapping phase
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()