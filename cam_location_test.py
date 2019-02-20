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
import time

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras.models import model_from_json
from vis.visualization import visualize_cam

import matplotlib.pyplot as plt

IMAGE_DIR = "GemImages"
TRAIN_DIR = "Train"
VALIDATE_DIR = "Validate"
TEST_DIR = "TEST"
POSSIBLE_SHAPES = ["Round","Heart","Baguette","Oval","Princess"]

IMAGE_DIM_X = 254
IMAGE_DIM_Y = 254

NUM_IMAGES = 5


if __name__ == '__main__':

    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMAGE_DIR)

    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    

    train_dir = os.path.join(base_dir, TRAIN_DIR)
    #print("Training images files in {0}".format(dir))

    val_dir = os.path.join(base_dir, VALIDATE_DIR)
    #print("Validation images files in {0}".format(dir))

    test_dir = os.path.join(base_dir, TEST_DIR)
    #print("Test images files in {0}".format(dir))

    # load json and create model
    json_file = open(os.path.join(script_dir, 'gem_classification_model_with_cam.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(script_dir, 'cnn_gem_shapes_with_cam.h5'))
    print("Loaded model from disk")

    loaded_model.compile(loss='mean_squared_error',     #mean_squared_error 
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc'])
    
    tests = []
    tests.append((0, "Baguette"))
    tests.append((1, "Heart"))
    tests.append((2, "Oval"))
    tests.append((3, "Princess"))
    tests.append((4, "Round"))

    classes = {'Baguette': 0, 'Heart': 1, 'Oval': 2, 'Princess': 3, 'Round': 4}
    value_list = []
    for name, index in classes.items():
        value_list.append((int(index), name))

    for test in tests:    
        load_dir = test_dir + '\\' + test[1]
        print("-------------------------------")
        print("Testing {0}s....". format(test[1]))
        
        file_list = os.listdir(load_dir)
        print("Test directory ({0}) contains {1} files.".format(load_dir, len(file_list)))
        file_name_list = []
        for file in file_list:
            file_name_list.append(os.path.join(load_dir , file)) 
        
        all_images = []

        #Pick 5 random images from the directory
        list_to_vis = []


        for i in range(0, NUM_IMAGES):
            list_to_vis.append(random.choice(file_name_list))

        for file in list_to_vis:
            img = cv2.imread(file)
            img= img/255.0
            all_images.append(img)
            print("Visualising image {0}".format(file))
        
        fig = plt.figure(figsize = (10, 10))
        
        for i in range(len(all_images)):                      
            for j in range(len(tests)):                
                heat_map = visualize_cam(loaded_model, 9, [j], all_images[i], backprop_modifier=None)
                fig.add_subplot(NUM_IMAGES, len(tests), (i * len(tests)) + j + 1)
                plt.axis('off')                
                plt.imshow(heat_map)
        fig.title =  test[1]
        plt.show()

