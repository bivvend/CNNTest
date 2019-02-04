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

IMAGE_DIR = "AngleTest"
TRAIN_DIR = "Train"
VALIDATE_DIR = "Validate"
TEST_DIR = "TEST"
POSSIBLE_SHAPES = ["Square","Triangle","Hexagon"]

ANGLE_THRESHOLD = 2.0

POSSIBLE_ANGLES = ["-30", "-25", "-20", "-10", "-5", "0", "5", "10", "15", "20", "25"]

Shape = "Hexagon"

IMAGE_DIM_X = 254
IMAGE_DIM_Y = 254


if __name__ == '__main__':

    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMAGE_DIR)

    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))    

    test_dir = os.path.join(base_dir, Shape  + "\\" +  TEST_DIR )
    print("Test images directories in {0}".format(test_dir))

    #load json and create model
    
    json_file = open(os.path.join(script_dir, 'angle_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(script_dir, 'cnn_angles.h5'))
    print("Loaded model from disk")

    loaded_model.compile(loss='mean_squared_error',     #mean_squared_error 
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc'])
    
    tests = []

    subfolders = [f.path for f in os.scandir(test_dir) if f.is_dir() ] 
    
    for folder in subfolders:
        tests.append((folder, folder.split('\\')[-1]))

    for test in tests:    
        load_dir = test[0]
        print("-------------------------------")
        print("Testing angles {0} in directory {1}". format(test[1], test[0]))
        
        file_list = os.listdir(load_dir)

        print("Test directory contains {0} files.".format(len(file_list)))
        file_name_list = []
        for file in file_list:
            file_name_list.append(os.path.join(load_dir , file)) 

        expected_rough_angle = test[1] # Value into which directory it was put
        
        all_images = []
        real_angle_values = []
        for file in file_name_list:
            img = cv2.imread(file)
            img= img/255.0
            all_images.append(img)
            real_angle_values.append(float(file.split()[-1].split('.')[0]))

        print("Number of images loaded = {0}".format(len(all_images)))
        

        start = time.time()
        x_train = np.array(all_images)
        results = loaded_model.predict(x_train)
        
        end = time.time()
        
        if Shape == "Hexagon":
            classes = {'-10': 0, '-20': 1, '-25': 2, '-30': 3, '-5': 4, '0': 5, '10': 6, '15': 7, '20': 8, '25': 9, '5': 10}

        value_list = []
        for name, index in classes.items():
            value_list.append((int(index), int(name)))

        print("Predicted {0} classes in {1} seconds".format(len(all_images), end - start))
        count = 0
        iter_count = 0
        mse = 0.0
        se = 0.0 
        for res in results:
            sum_vals = 0.0
            for i in range(0, len(value_list)):
                sum_vals += res[i] * float(value_list[i][1])
            se += math.pow (sum_vals - real_angle_values[iter_count], 2) 
            print("Predicted angle of {0}. Real angle {1}".format(sum_vals, real_angle_values[iter_count]))           
            if abs(sum_vals - real_angle_values[iter_count]) > ANGLE_THRESHOLD:
                count += 1
            iter_count += 1
        mse = se / iter_count
        print("{0} errors detected with a mse of {1}".format(count, mse))
        