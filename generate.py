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

IMAGE_DIR = "Images"
TRAIN_DIR = "Train"
VALIDATE_DIR = "Validate"
TEST_DIR = "TEST"

IMAGE_DIM_X = 254
IMAGE_DIM_Y = 254

NUM_IMAGES = 2000

def generate_image_bath(dir, number):
    if os.path.isdir(dir) != True:
        os.mkdir(dir)
    
    imageName = "Shape "
    for i in range(number):        
        shape = random.choice(["square","triangle","hexagon"])
        image = np.zeros([IMAGE_DIM_X, IMAGE_DIM_Y, 3], dtype=np.uint8)
        buffer = 40
        y = random.randint(buffer, IMAGE_DIM_Y - buffer - 1)
        x = random.randint(buffer, IMAGE_DIM_X - buffer - 1)
        angle = random.randint(0, 360)
        # Size
        s = random.randint(buffer, int(IMAGE_DIM_X/4)) 
        dims = (x, y, s, angle)     
        image = draw_shape(image, shape, dims)
        imageName = "Shape {0} {1} {2} {3} {4} {5}.bmp".format(shape, i, x , y , s, angle)
        filepath = os.path.join(dir, imageName)
        cv2.imwrite(filepath, image) 

def rotate( vec, angle):

    x,y = vec
    rads = np.radians(angle)

    new_x = math.cos(rads)* x - math.sin(rads) * y
    new_y = math.sin(rads) * x + math.cos(rads) * y

    return (new_x, new_y)

def draw_shape(image, shape, dims):
    """Draws a shape from the given specs."""
    # Get the center x, y and the size s
    x, y, s, angle = dims
    colour = (255,255,255)
    if shape == 'square':
        #cv2.rectangle(image, (x-s, y-s), (x+s, y+s), colour, -1)
        x1 = -1*s
        y1 = -1*s
        x2 = s
        y2 = -1*s
        x3 = s
        y3 = s
        x4 = -1*s
        y4 = s        
        
        x1a, y1a = rotate((x1, y1), angle)
        x2a, y2a = rotate((x2, y2), angle)
        x3a, y3a = rotate((x3, y3), angle)
        x4a, y4a = rotate((x4, y4), angle)

        #print("{0},{1} to {2},{3}".format(x1, y1, x1a, y1a))
        points = np.array([[(x1a, y1a), (x2a, y2a), (x3a, y3a), (x4a, y4a)]], dtype=np.int32)
        
    elif shape == "triangle":
        x1 = 0  
        y1 = -s
        x2 = -1 * s/math.sin(math.radians(60))
        y2 = s
        x3 = s/math.sin(math.radians(60))
        y3 = s
        
        x1a, y1a = rotate((x1, y1), angle)
        x2a, y2a = rotate((x2, y2), angle)
        x3a, y3a = rotate((x3, y3), angle)

        points = np.array([[(x1a, y1a),
                            (x2a, y2a),
                            (x3a, y3a),
                            ]], dtype=np.int32)

    elif shape == "hexagon":
        x1 = int(-1.0 * float(s))  
        y1 = 0
        x2 = int(-0.5 * float(s)) 
        y2 = int(float(s) * math.sqrt(3)/2)
        x3 = int(0.5 * float(s))
        y3 = int(float(s) * math.sqrt(3)/2)
        x4 = int(1.0 *float(s))
        y4 = 0
        x5= int(0.5 * float(s))
        y5 = int(float(s) * -1.0 * math.sqrt(3)/2)
        x6= int(-0.5 * float(s))
        y6 = int(float(s) * -1.0 * math.sqrt(3)/2)

        x1a, y1a = rotate((x1, y1), angle)
        x2a, y2a = rotate((x2, y2), angle)
        x3a, y3a = rotate((x3, y3), angle)
        x4a, y4a = rotate((x4, y4), angle)
        x5a, y5a = rotate((x5, y5), angle)
        x6a, y6a = rotate((x6, y6), angle)
        points = np.array([[(x1a, y1a),
                            (x2a, y2a),
                            (x3a, y3a),
                            (x4a, y4a),
                            (x5a, y5a),
                            (x6a, y6a),
                            ]], dtype=np.int32)  


    point_shape = points.shape
    #print(point_shape)
    offset = np.zeros((1,0,2), dtype=np.int32)

    for i in range(point_shape[1]):
        add = np.array([[(x,y)]])
        #print(add.shape)
        offset = np.append(offset, add)
    
    shaped_offset = offset.reshape(point_shape)
    #add offset
    new_points = points + shaped_offset
    #print(points)
    #print(new_points)                            
    cv2.fillPoly(image, new_points, colour)
    return image   

if __name__ == '__main__':

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMAGE_DIR)
    if os.path.isdir(base_dir) != True:
        os.mkdir(base_dir)

    dir = os.path.join(base_dir, TRAIN_DIR)
    print("Generating training images files in {0}".format(dir))
    generate_image_bath(dir, int(NUM_IMAGES * 0.7))

    dir = os.path.join(base_dir, VALIDATE_DIR)
    print("Generating validation images files in {0}".format(dir))
    generate_image_bath(dir, int(NUM_IMAGES * 0.2))

    dir = os.path.join(base_dir, TEST_DIR)
    print("Generating validation images files in {0}".format(dir))
    generate_image_bath(dir, int(NUM_IMAGES * 0.1))
        
