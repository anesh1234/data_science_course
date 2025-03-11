from ast import main
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image


def getClassNames(train_labels, test_labels, valid_labels):
    allConcat = pd.concat([train_labels, test_labels, valid_labels])
    unique_classes = allConcat['class'].unique()
    print(unique_classes)

def plotData(train_labels, test_labels, valid_labels):
    # Define image folder
    train_images_folder = Path("data/radiography/train")

    # Function to plot bounding boxes on images
    def plot_bounding_boxes(image_path, bboxes, labels):
        # Open the image
        image = Image.open(image_path)
        
        # Create a plot
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        
        # Plot each bounding box
        for bbox, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            width = xmax - xmin
            height = ymax - ymin
            
            # Create a rectangle patch
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
            
            # Add the patch to the Axes
            ax.add_patch(rect)
            
            # Add label text
            plt.text(xmin, ymin - 10, label, color='red', fontsize=12, backgroundcolor='white')
        plt.show()

    # Get unique filenames from the CSV file
    unique_filenames = train_labels['filename'].unique()
    print(unique_filenames)

    # Plot bounding boxes for the first 10 images
    for filename in unique_filenames[:20]:
        image_path = train_images_folder / filename
        
        # Get bounding boxes and labels for the current image
        bboxes = train_labels[train_labels['filename'] == filename][['xmin', 'ymin', 'xmax', 'ymax']].values
        labels = train_labels[train_labels['filename'] == filename]['class'].values

        print("\nBBoxes: ", bboxes)
        print("\nLabels: ", labels)
        
        # Plot bounding boxes on the image
        plot_bounding_boxes(image_path, bboxes, labels)
    
if __name__ == "__main__":
    # Load the CSV files
    train_labels = pd.read_csv('data/radiography/train/_annotations.csv')
    test_labels = pd.read_csv('data/radiography/test/_annotations.csv')
    valid_labels = pd.read_csv('data/radiography/valid/_annotations.csv')

    getClassNames(train_labels, test_labels, valid_labels)
    #plotData(train_labels, test_labels, valid_labels)