import multiprocessing
from ultralytics import YOLO

def main():
    # Load a YOLOv11 Nano model if it exists, downloads otherwise
    model = YOLO('yolo11n.pt')
    results = model.tune(data= "datasets/final/data.yaml",
                         iterations= 30,
                         epochs= 5,
                         project= 'YOLO/tuning_runs',
                         device= 'cuda'
                         )

    print(results)

# Idiom to ensure that a new process cannot start before the
# current process has finished its bootstrapping phase
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()