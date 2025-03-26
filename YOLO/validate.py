from ultralytics import YOLO
import multiprocessing
import csv

def main(train_id, config):
    
    # Load a model
    model = YOLO(f"YOLO/{config}/train_id_{train_id}/weights/best.pt")

    # Perform the validation
    results = model.val(project = f'YOLO/{config}/val_id_{train_id}')

    # Print specific metrics from the validation result
    print("\nClass indices with average precision:", results.ap_class_index)
    print("\nAverage precision for all classes:", results.box.all_ap)
    print("\nAverage precision:", results.box.ap)                                      # Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.
    print("\nAverage precision at IoU=0.50:", results.box.ap50)                        # Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.
    print("\nClass-specific results:", results.box.class_result)
    print("\nF1 score:", results.box.f1)
    print("\nMean average precision 0.5-0.95:", results.box.map)
    print("\nMean average precision at IoU=0.50:", results.box.map50)

    # Combine the metrics into an ordered list
    measurements = [f'{config}', results.box.map50, results.box.map]


    # Define the title for the CSV file
    title = f"{config}_id_{train_id}_val"

    # Define the column headers
    #headers = ["Model", "mAP50", "mAP50-95", "BBox Loss", "Class Loss", "Epochs"]
    headers = ["Model", "mAP50", "mAP50-95"]

    # Create and write to the CSV file
    with open(f"{title}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the headers
        writer.writerow(measurements)  # Write the measurements


# Idiom to ensure that a new process cannot start before the
# current process has finished its bootstrapping phase
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Possible configs:
    # 'npt_nfrozen' - Not pre-trained and no frozen layers
    # 'pt_frozen' - Pre-trained and frozen CNN backbone
    # 'pt_nfrozen' - Pre-trained and no frozen layers

    train_id = '3_part2'
    config = 'pt_nfrozen'
    main(train_id, config)