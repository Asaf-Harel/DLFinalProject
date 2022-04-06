import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
from PIL import Image
import random

import tensorflow as tf
from tensorflow.keras import layers, models, losses


def get_data():
    """Get the train and test data for training

    Returns:
        Tuple: Tuple of the x_train, y_train and x_test, y_test
    """
    data_path = path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    CLASS_NAMES = os.listdir(data_path)  # Get the class names from the directories names
    classes = {}

    for i in range(len(CLASS_NAMES)):
        classes[CLASS_NAMES[i]] = i

    X = []
    Y = []
    for class_name in CLASS_NAMES:
        for file in os.listdir(path.join(data_path, class_name)):
            if '.png' in file:
                image_path = path.join(data_path, class_name, file)
                X.append(np.array(Image.open(image_path).resize((220, 220))))  # Open the images and convert them into 220x220
                Y.append(classes[class_name])  # Append the correspond class name

    X = np.array(X)
    Y = np.array(Y)

    n = round(X.shape[0] * 0.7)

    X_train, Y_train = X[:n], Y[:n]
    temp = list(zip(X_train, Y_train))
    random.shuffle(temp)
    X_train, Y_train = zip(*temp)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X_test, Y_test = X[n:], Y[n:]
    temp = list(zip(X_test, Y_test))
    random.shuffle(temp)
    X_test, Y_test = zip(*temp)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return (X_train, Y_train), (X_test, Y_test)


def create_model(weights_path: str):
    """Create the AlexNet model architecture

    Args:
        weights_path (str): The saved weights path

    Returns:
        tensorflow.keras.models.Sequential: The model
    """
    model = models.Sequential()
    model.add(
        layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=(224, 224, 3)))
    model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))
    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    model.load_weights(weights_path)

    return model


def normalize_image(X: np.ndarray):
    """Normalize image values and add padding

    Args:
        X (np.ndarray): The images array

    Returns:
        tensorflow.Tensor: normalized images
    """
    return tf.pad(X, [[0, 0], [2, 2], [2, 2], [0, 0]]) / 255

