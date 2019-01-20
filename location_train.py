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


if __name__ == '__main__':