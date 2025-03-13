import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import config


def getClassNames(train_labels, test_labels, valid_labels):
    allConcat = pd.concat([train_labels, test_labels, valid_labels])
    unique_classes = allConcat['class'].unique()
    print(unique_classes)

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

def plotData(train_labels, test_labels, valid_labels):
    # Define image folder
    train_images_folder = Path("data/radiography/train")

    # Get unique filenames from the CSV file
    unique_filenames = train_labels['filename'].unique()

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
    
def plotTwoImagesWithBoundingBoxes(image1: Image.Image, image2: Image.Image, bbox1: pd.DataFrame, bbox2: pd.DataFrame, img_titles: list[str]):
    """
    Utility function to plot 2 different images in 2 editions: original and with bboxes.

    Image1 will be put in index [0,0] and [0,1], assuming titles img_titles[0 and 1]

    image2 will be put in index [1,0] and [1,1], assuming titles img_titles[2 and 3]

    The function currently assumes that labels/classes are ints and thus remaps them
    """

    # Plotting the results versus the original
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Plot the new image WITHOUT bounding boxes on the top left
    axes[0,0].imshow(image1)
    axes[0,0].set_title(img_titles[0])
    axes[0,0].axis("off")

    # Plot the new image WITH bounding boxes on the top right
    axes[0,1].imshow(image1)
    for _, bbox in bbox1.iterrows():
        x_min, y_min, x_max, y_max, class_id = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=2, edgecolor='r', facecolor='none')
        axes[0,1].add_patch(rect)
        axes[0,1].text(x_min, y_min - 10, f"{config.index_to_class[class_id]}", color='r', fontsize=12)
    axes[0,1].set_title(img_titles[1])
    axes[0,1].axis("off")

    #########################################################################################

    # Plot the original image WITHOUT bounding boxes on the bottom right
    axes[1,0].imshow(image2)
    axes[1,0].set_title(img_titles[2])
    axes[1,0].axis("off")

    # Plot the original image WITH bounding boxes on the bottom right
    axes[1,1].imshow(image2)
    for _, bbox in bbox2.iterrows():
        x_min, y_min, x_max, y_max, class_id = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=2, edgecolor='r', facecolor='none')
        axes[1,1].add_patch(rect)
        axes[1,1].text(x_min, y_min - 10, f"{config.index_to_class[class_id]}", color='r', fontsize=12)
    axes[1,1].set_title(img_titles[3])
    axes[1,1].axis("off")

    plt.show()

if __name__ == "__main__":
    # Load the CSV files
    train_labels = pd.read_csv('data/radiography/train/_annotations.csv')
    test_labels = pd.read_csv('data/radiography/test/_annotations.csv')
    valid_labels = pd.read_csv('data/radiography/valid/_annotations.csv')

    #getClassNames(train_labels, test_labels, valid_labels)
    #plotData(train_labels, test_labels, valid_labels)