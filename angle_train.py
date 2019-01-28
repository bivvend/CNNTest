import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers

import matplotlib.pyplot as plt

IMAGE_DIR = "AngleTest"
SHAPE = "Hexagon"
TRAIN_DIR = "Train"
VALIDATE_DIR = "Validate"
TEST_DIR = "Test"
POSSIBLE_ANGLES = ["-30", "-25", "-20", "-10", "-5", "0", "5", "10", "15", "20", "25"]

IMAGE_DIM_X = 254
IMAGE_DIM_Y = 254


if __name__ == '__main__':

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMAGE_DIR + "\\" + SHAPE)
    
    train_dir = os.path.join(base_dir, TRAIN_DIR)
    print("Training images files in {0}".format(dir))


    val_dir = os.path.join(base_dir, VALIDATE_DIR)
    print("Validation images files in {0}".format(dir))

    test_dir = os.path.join(base_dir, TEST_DIR)
    print("Test images files in {0}".format(dir))

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_DIM_X, IMAGE_DIM_Y),
        batch_size=20,
        class_mode='categorical')
    
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(IMAGE_DIM_X, IMAGE_DIM_Y),
        batch_size=20,
        class_mode='categorical')
    
    for data_batch, labels_batch in train_generator:
        print("Data batch shape:", data_batch.shape)
        print("Labels batch shape:", labels_batch.shape)
        break

    #Build convnet
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (IMAGE_DIM_X, IMAGE_DIM_Y, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(len(POSSIBLE_ANGLES), activation='softmax'))

    print(model.summary())

    model.compile(loss='mean_squared_error',     #mean_squared_error #categorical_crossentropy
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc']) 

    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 50,
        validation_data = validation_generator,
        validation_steps = 50)  
    
    model_json = model.to_json()
    with open("angle_model.json", "w") as json_file:
        json_file.write(model_json)    
    model.save_weights("cnn_angles.h5")
    print("Saved model to disk")
    
    acc = history.history['acc']
    val_acc  = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label = "Train acc")
    plt.plot(epochs, val_acc, 'b', label = "Validation acc")
    plt.title("Training and validation acc")
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label = "Train loss")
    plt.plot(epochs, val_loss, 'b', label = "Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.show()

