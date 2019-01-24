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

IMAGE_DIR = "Images"
TRAIN_DIR = "Train"
VALIDATE_DIR = "Validate"
TEST_DIR = "TEST"
POSSIBLE_SHAPES = ["Square","Triangle","Hexagon"]

IMAGE_DIM_X = 254
IMAGE_DIM_Y = 254

def data_gen(image_dir, batch_size_in):
    """
    Generator to yield batches of two inputs (per sample) with shapes top_dim and 
    bot_dim along with their labels.
    """
    batch_size = batch_size_in
    image_file_list = os.listdir(image_dir)
    num_images = len(image_file_list)    
    while True:        
        image_batch = []
        batch_labels = []
        for i in range(batch_size):
            # Create random arrays
            rand = np.random.randint(0, len(image_file_list))
            full_path = os.path.join(image_dir, image_file_list[rand])
            img = cv2.imread(full_path, 0)
            img= cv2.resize(img, (IMAGE_DIM_X, IMAGE_DIM_Y))
            img = img.reshape(IMAGE_DIM_X*IMAGE_DIM_Y)
            img= img.astype('float32')/255.0
            image_batch.append(img)
            # Set a label
            splits = image_file_list[rand].split()
            angle =  float(splits[6].split('.')[0])/30.0
            xcoord = float(splits[3])/ 254.0
            ycoord = float(splits[4])/ 254.0
            batch_labels.append(xcoord)

        yield np.array(image_batch), np.array(batch_labels)

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
    
    
    #Build convnet
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (IMAGE_DIM_X, IMAGE_DIM_Y, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3,3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3,3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(128, (3,3), activation='relu'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dense(1))  #angle  

    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape = (IMAGE_DIM_X * IMAGE_DIM_Y,)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))   

    print(model.summary())

    model.compile(loss='mean_squared_error',     #mean_squared_error #categorical_crossentropy
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['acc']) 
    
    train_generator = data_gen(train_dir, 100)
    valid_generator = data_gen(val_dir, 100)

    history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=100,
                    validation_data=valid_generator,
                    validation_steps=50,
                    epochs=20)