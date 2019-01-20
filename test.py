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
from keras.models import model_from_json

import matplotlib.pyplot as plt

IMAGE_DIR = "Images"
TRAIN_DIR = "Train"
VALIDATE_DIR = "Validate"
TEST_DIR = "TEST"
POSSIBLE_SHAPES = ["Square","Triangle","Hexagon"]

IMAGE_DIM_X = 254
IMAGE_DIM_Y = 254


if __name__ == '__main__':

    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMAGE_DIR)

    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    

    train_dir = os.path.join(base_dir, TRAIN_DIR)
    print("Training images files in {0}".format(dir))

    val_dir = os.path.join(base_dir, VALIDATE_DIR)
    print("Validation images files in {0}".format(dir))

    test_dir = os.path.join(base_dir, TEST_DIR)
    print("Test images files in {0}".format(dir))

    # load json and create model
    json_file = open(os.path.join(script_dir, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(script_dir, 'cnn_shapes.h5'))
    print("Loaded model from disk")

    loaded_model.compile(loss='mean_squared_error',     #mean_squared_error 
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['acc'])

    file_list = os.listdir(test_dir+ "\Hexagon")
    print("Test directory ({0}) contains {1} files.".format(test_dir, len(file_list)))
    
    print(cv2.imread(file_list[0]))
    #loaded_model.predict(cv2.imread(file_list[0]))