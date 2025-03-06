import numpy as np
import os
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

print(tf.__version__)

BATCH_SIZE = 32
IMG_HEIGHT = 512
IMG_WIDTH = 256
DATA_DIR = Path("data/radiography")

# Define CSV file's paths
train_csv_path = Path("data/radiography/train/_annotations.csv")
valid_csv_path = Path("data/radiography/valid/_annotations.csv")
test_csv_path = Path("data/radiography/test/_annotations.csv")

# Define image folders
train_images_folder = Path("data/radiography/train")
valid_images_folder = Path("data/radiography/valid")
test_images_folder = Path("data/radiography/test")

# Read CSV files
train_annotations = pd.read_csv(train_csv_path)
valid_annotations = pd.read_csv(valid_csv_path)
test_annotations = pd.read_csv(test_csv_path)

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Load and preprocess images and annotations
def load_images_and_annotations(annotations, images_folder):
    images = []
    labels = []
    bboxes = []
    for _, row in annotations.iterrows():
        image_path = images_folder / row['filename']
        image = load_and_preprocess_image(image_path)
        images.append(image)
        
        # Extract bounding box and class label
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        bboxes.append(bbox)
        labels.append(row['class'])
        
    return np.array(images), np.array(labels), np.array(bboxes)

# Load images, labels, and bounding boxes
train_images, train_labels, train_bboxes = load_images_and_annotations(train_annotations, train_images_folder)
valid_images, valid_labels, valid_bboxes = load_images_and_annotations(valid_annotations, valid_images_folder)
test_images, test_labels, test_bboxes = load_images_and_annotations(test_annotations, test_images_folder)

print(f'Train images shape: {train_images.shape}')
print(f'Validation images shape: {valid_images.shape}')
print(f'Test images shape: {test_images.shape}')
print(f'Train labels shape: {train_labels.shape}')
print(f'Train bounding boxes shape: {train_bboxes.shape}')


# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, bboxes, labels):
    image = Image.fromarray((image * 255).astype(np.uint8))  # Convert back to uint8
    draw = ImageDraw.Draw(image)
    for bbox, label in zip(bboxes, labels):
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), label, fill="red")
    return image

# Visualize a few images with bounding boxes
def visualize_images_with_bboxes(images, bboxes, labels, num_images=5):
    plt.figure(figsize=(IMG_WIDTH/10, IMG_HEIGHT/10))
    image_with_bboxes = draw_bounding_boxes(images[0], [bboxes[0]], [labels[0]])
    plt.imshow(image_with_bboxes)
    plt.axis('off')
    plt.show()

# Visualize some training images with bounding boxes
visualize_images_with_bboxes(train_images, train_bboxes, train_labels)