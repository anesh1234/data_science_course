from ultralytics import YOLO
import multiprocessing

def main(train_id, config):
    
    # Load a model
    model = YOLO("YOLO/pt_frozen/train_id_2/weights/best.pt")  # load a custom model

    results = model.val(project = f'YOLO/{config}/val_id_{train_id}')

    # Print specific metrics
    print("Class indices with average precision:", results.ap_class_index)
    print("Average precision for all classes:", results.box.all_ap)
    print("Average precision:", results.box.ap)
    print("Average precision at IoU=0.50:", results.box.ap50)
    print("Class indices for average precision:", results.box.ap_class_index)
    print("Class-specific results:", results.box.class_result)
    print("F1 score:", results.box.f1)
    print("F1 score curve:", results.box.f1_curve)
    print("Overall fitness score:", results.box.fitness)
    print("Mean average precision:", results.box.map)
    print("Mean average precision at IoU=0.50:", results.box.map50)
    print("Mean average precision at IoU=0.75:", results.box.map75)
    print("Mean average precision for different IoU thresholds:", results.box.maps)
    print("Mean results for different metrics:", results.box.mean_results)
    print("Mean precision:", results.box.mp)
    print("Mean recall:", results.box.mr)
    print("Precision:", results.box.p)
    print("Precision curve:", results.box.p_curve)
    print("Precision values:", results.box.prec_values)
    print("Specific precision metrics:", results.box.px)
    print("Recall:", results.box.r)
    print("Recall curve:", results.box.r_curve)


# Idiom to ensure that a new process cannot start before the
# current process has finished its bootstrapping phase
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Possible configs:
    # 'npt_nfrozen' - Not pre-trained and no frozen layers
    # 'pt_frozen' - Pre-trained and frozen CNN backbone
    # 'pt_nfrozen' - Pre-trained and no frozen layers

    train_id = 2
    config = 'pt_nfrozen'
    main(train_id, config)