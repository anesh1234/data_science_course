import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import pandas as pd

IMG_HEIGHT = 150
IMG_WIDTH = 150

train_labels = pd.read_csv('data/radiography/train/_annotations.csv')
test_labels = pd.read_csv('data/radiography/test/_annotations.csv')
valid_labels = pd.read_csv('data/radiography/valid/_annotations.csv')



# # https://keras.io/api/applications/resnet/#resnet50-function
# base_model = keras.applications.ResNet50(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
#     name="resnet50",
# )

# # Freeze the base_model
# base_model.trainable = False

# # Create new model on top
# inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# # Pre-trained Xception weights requires that input be scaled
# # from (0, 255) to a range of (-1., +1.), the rescaling layer
# # outputs: `(inputs * scale) + offset`
# scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
# x = scale_layer(inputs)

# # The base model contains batchnorm layers. We want to keep them in inference mode
# # when we unfreeze the base model for fine-tuning, so we make sure that the
# # base_model is running in inference mode here.
# x = base_model(x, training=False)
# x = keras.layers.GlobalAveragePooling2D()(x)
# x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
# outputs = keras.layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)

# model.summary(show_trainable=True)