import tensorflow as tf
import os
import requests
import json
import matplotlib.pyplot as plt

# Load the model and dataset
model = tf.keras.models.load_model('testModel.keras')
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Show the model architecture
model.summary()

# Inference
prediction = model.predict(x_test[:1])

# Show sample predictions
fig, axes = plt.subplots(2, 2, figsize=(2, 2))
i, ax = enumerate(axes)
fig.imshow(x_test[0], cmap='gray')
ax.set_title(f"Pred: {prediction[0].argmax()}")
ax.axis("off")
plt.show()