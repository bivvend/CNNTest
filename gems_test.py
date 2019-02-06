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

import matplotlib.pyplot as plt

IMAGE_DIR = "GemImages"
TRAIN_DIR = "Train"
VALIDATE_DIR = "Validate"
TEST_DIR = "TEST"
POSSIBLE_SHAPES = ["Round","Heart","Baguette","Oval","Princess"]

IMAGE_DIM_X = 254
IMAGE_DIM_Y = 254


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
    json_file = open(os.path.join(script_dir, 'gem_classification_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(script_dir, 'cnn_gem_shapes.h5'))
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
        for file in file_name_list:
            img = cv2.imread(file)
            img= img/255.0
            all_images.append(img)

        start = time.time()
        x_train = np.array(all_images)
        classes = loaded_model.predict_classes(x_train)
        end = time.time()
        
        print("Predicted {0} classes in {1} seconds".format(len(all_images), end - start))
        count = 0
        for res in classes:
            if res != test[0]:
                count += 1
        print("{0} errors detected ".format(count))
        