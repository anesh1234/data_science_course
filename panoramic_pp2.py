import numpy as np
import os
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
#from keras._tf_keras.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Define CSV file's paths
train_csv_path = Path("data/radiography/train/_annotations.csv")
valid_csv_path = Path("data/radiography/valid/_annotations.csv")
test_csv_path = Path("data/radiography/test/_annotations.csv")

# Define image folders
train_images_folder = Path("data/radiography/train")
valid_images_folder = Path("data/radiography/valid")
test_images_folder = Path("data/radiography/test")

data = tf.keras.utils.image_dataset_from_directory('data')
print("\nDATA: ",data,"\n")