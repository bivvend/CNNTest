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
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers

import matplotlib.pyplot as plt

IMAGE_DIR = "Images"
TRAIN_DIR = "Train"
VALIDATE_DIR = "Validate"
TEST_DIR = "TEST"
POSSIBLE_SHAPES = ["Square","Triangle","Hexagon"]

IMAGE_DIM_X = 254
IMAGE_DIM_Y = 254


if __name__ == '__main__':

    shape_name = "Hexagon"

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMAGE_DIR)
    
    train_dir = os.path.join(base_dir, TRAIN_DIR + "\\" + shape_name )
    print("Training images files in {0}".format(dir))


    val_dir = os.path.join(base_dir, VALIDATE_DIR + "\\" + shape_name)
    print("Validation images files in {0}".format(dir))

    test_dir = os.path.join(base_dir, TEST_DIR + "\\" + shape_name)
    print("Test images files in {0}".format(dir))

    #build data
    
    file_list = os.listdir(train_dir)
    print("Train directory ({0}) contains {1} files.".format(train_dir, len(file_list)))
    file_name_list = []
    for file in file_list:
        file_name_list.append(os.path.join(train_dir , file)) 
    
    #https://medium.com/@vijayabhaskar96/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1
    # df=pd.read_csv(r".\train.csv")
    # datagen=ImageDataGenerator(rescale=1./255)
    # train_generator=datagen.flow_from_dataframe(dataframe=df, directory=".\train_imgs", x_col="id", y_col="label", class_mode="categorical", target_size=(32,32), batch_size=32)
    
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
    model.add(layers.Dense(3))  # x, y, angle  

    print(model.summary())

    model.compile(loss='mean_squared_error',     #mean_squared_error #categorical_crossentropy
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc']) 
    
    # STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    # STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    # model.fit_generator(generator=train_generator,
    #                 steps_per_epoch=STEP_SIZE_TRAIN,
    #                 validation_data=valid_generator,
    #                 validation_steps=STEP_SIZE_VALID,
    #                 epochs=10)