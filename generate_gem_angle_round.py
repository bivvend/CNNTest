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

IMAGE_DIR = "GemAngle"
SHAPE = "Round"
TRAIN_DIR = "Train"
VALIDATE_DIR = "Validate"
TEST_DIR = "Test"
POSSIBLE_ANGLES = ["0", "5", "10", "15", "20", "25", "30", "35", "40"]  # Octagon so 0 -> 45

IMAGE_DIM_X = 254
IMAGE_DIM_Y = 254
NUM_IMAGES = 40000

def generate_image_batch(dir, number, angles_are_continuous):
    if os.path.isdir(dir) != True:
        os.mkdir(dir)
    for s in POSSIBLE_ANGLES:
        angle_path = os.path.join(dir, s)
        if os.path.isdir(angle_path) != True:
            os.mkdir(angle_path) 
    
    image_name = "Shape "
    for i in range(number): 
        angle_name = random.choice(POSSIBLE_ANGLES)        
        angle = int(angle_name)
        if(angles_are_continuous):
            angle = angle + random.uniform(-2.25,2.25)            
        background_colour = 100
        image = np.zeros([IMAGE_DIM_X, IMAGE_DIM_Y, 3], dtype=np.uint8) + background_colour
        shape = SHAPE
        buffer = 40
        y = random.randint(buffer, IMAGE_DIM_Y - buffer - 1)
        x = random.randint(buffer, IMAGE_DIM_X - buffer - 1)
        # Size
        s = random.randint(buffer, int(IMAGE_DIM_X/4)) 
        dims = (x, y, s, angle)     
        image = draw_shape(image, shape, dims)
        image_name = "Shape {0} {1} {2} {3} {4} {5}.bmp".format(shape, i, x , y , s, str(round(angle,4)))
        save_dir = os.path.join(dir, angle_name)
        
        filepath = os.path.join(save_dir, image_name)
        #Make sub directory for generator
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
    colour = (220,220,220)
    if shape == 'Baguette':
        #cv2.rectangle(image, (x-s, y-s), (x+s, y+s), colour, -1)
        
        x1 = -1*s
        y1 = -0.55*s
        x2 = s
        y2 = -0.55*s
        x3 = s
        y3 = 0.55*s
        x4 = -1*s
        y4 = 0.55*s        
        
        x1a, y1a = rotate((x1, y1), angle)
        x2a, y2a = rotate((x2, y2), angle)
        x3a, y3a = rotate((x3, y3), angle)
        x4a, y4a = rotate((x4, y4), angle)

        #print("{0},{1} to {2},{3}".format(x1, y1, x1a, y1a))
        points = np.array([[(x1a, y1a), (x2a, y2a), (x3a, y3a), (x4a, y4a)]], dtype=np.int32)
        
    elif shape == "Heart":
        
        x1 = 0.0  * s
        y1 = -0.817 * s
        x2 = -0.786 * s
        y2 = -0.573 * s
        x3 = -0.821 * s
        y3 = -0.153 * s
        x4 = -0.384 * s 
        y4 = 0.668 * s
        x5 = 0.0 *s
        y5 = 1.0 * s
        x6 = 0.384 * s
        y6 = 0.668 * s
        x7 = 0.821 * s 
        y7 = -0.153 * s
        x8 = 0.786 * s
        y8 = -0.573 * s
        x9 = 0.0 * s
        y9 = -0.817 * s
        
        x1a, y1a = rotate((x1, y1), angle)
        x2a, y2a = rotate((x2, y2), angle)
        x3a, y3a = rotate((x3, y3), angle)
        x4a, y4a = rotate((x4, y4), angle)
        x5a, y5a = rotate((x5, y5), angle)
        x6a, y6a = rotate((x6, y6), angle)
        x7a, y7a = rotate((x7, y7), angle)
        x8a, y8a = rotate((x8, y8), angle)
        x9a, y9a = rotate((x9, y9), angle)

        points = np.array([[(x1a, y1a),
                            (x2a, y2a),
                            (x3a, y3a),
                            (x4a, y4a),
                            (x5a, y5a),
                            (x6a, y6a),
                            (x7a, y7a),
                            (x8a, y8a),
                            (x9a, y9a)
                            ]], dtype=np.int32)

    elif shape == "Round":
        
        x1 = 0.383  * s
        y1 = -0.924 * s
        x2 = -0.383 * s
        y2 = -0.924 * s
        x3 = -0.924 * s
        y3 = -0.383 * s
        x4 = -0.924 * s 
        y4 = 0.383 * s
        x5 = -0.383 *s
        y5 = 0.924 * s
        x6 = 0.383 * s
        y6 = 0.924 * s
        x7 = 0.924 * s 
        y7 = 0.383 * s
        x8 = 0.924 * s
        y8 = -0.383 * s
        x9 = 0.383 * s
        y9 = -0.924 * s
        
        x1a, y1a = rotate((x1, y1), angle)
        x2a, y2a = rotate((x2, y2), angle)
        x3a, y3a = rotate((x3, y3), angle)
        x4a, y4a = rotate((x4, y4), angle)
        x5a, y5a = rotate((x5, y5), angle)
        x6a, y6a = rotate((x6, y6), angle)
        x7a, y7a = rotate((x7, y7), angle)
        x8a, y8a = rotate((x8, y8), angle)
        x9a, y9a = rotate((x9, y9), angle)

        points = np.array([[(x1a, y1a),
                            (x2a, y2a),
                            (x3a, y3a),
                            (x4a, y4a),
                            (x5a, y5a),
                            (x6a, y6a),
                            (x7a, y7a),
                            (x8a, y8a),
                            (x9a, y9a)
                            ]], dtype=np.int32) 

    elif shape == "Princess":
        
        x1 = 0.85  * s
        y1 = -0.85 * s
        x2 = 1.0 * s
        y2 = 0.0 * s
        x3 = 0.85 * s
        y3 = 0.85 * s
        x4 = 0.0 * s 
        y4 = 1.0 * s
        x5 = -0.85 *s
        y5 = 0.85 * s
        x6 = -1.0 * s
        y6 = 0.0 * s
        x7 = -0.85 * s 
        y7 = -0.85 * s
        x8 = 0.0 * s
        y8 = -1.0 * s
        x9 = 0.85 * s
        y9 = -0.85 * s
        
        x1a, y1a = rotate((x1, y1), angle)
        x2a, y2a = rotate((x2, y2), angle)
        x3a, y3a = rotate((x3, y3), angle)
        x4a, y4a = rotate((x4, y4), angle)
        x5a, y5a = rotate((x5, y5), angle)
        x6a, y6a = rotate((x6, y6), angle)
        x7a, y7a = rotate((x7, y7), angle)
        x8a, y8a = rotate((x8, y8), angle)
        x9a, y9a = rotate((x9, y9), angle)

        points = np.array([[(x1a, y1a),
                            (x2a, y2a),
                            (x3a, y3a),
                            (x4a, y4a),
                            (x5a, y5a),
                            (x6a, y6a),
                            (x7a, y7a),
                            (x8a, y8a),
                            (x9a, y9a)
                            ]], dtype=np.int32) 

    elif shape == "Oval":
        
        x1 = 0.7  * s
        y1 = -0.5 * s
        x2 = 1.0 * s
        y2 = 0.0 * s
        x3 = 0.7 * s
        y3 = 0.5 * s
        x4 = 0.0 * s 
        y4 = 0.7 * s
        x5 = -0.7 *s
        y5 = 0.5 * s
        x6 = -1.0 * s
        y6 = 0.0 * s
        x7 = -0.7 * s 
        y7 = -0.5 * s
        x8 = 0.0 * s
        y8 = -0.7 * s
        x9 = 0.7 * s
        y9 = -0.5 * s
        
        x1a, y1a = rotate((x1, y1), angle)
        x2a, y2a = rotate((x2, y2), angle)
        x3a, y3a = rotate((x3, y3), angle)
        x4a, y4a = rotate((x4, y4), angle)
        x5a, y5a = rotate((x5, y5), angle)
        x6a, y6a = rotate((x6, y6), angle)
        x7a, y7a = rotate((x7, y7), angle)
        x8a, y8a = rotate((x8, y8), angle)
        x9a, y9a = rotate((x9, y9), angle)

        points = np.array([[(x1a, y1a),
                            (x2a, y2a),
                            (x3a, y3a),
                            (x4a, y4a),
                            (x5a, y5a),
                            (x6a, y6a),
                            (x7a, y7a),
                            (x8a, y8a),
                            (x9a, y9a)
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

    #draw dark background circle
    cv2.circle(image,(x,y), int(s * 1.2), (0,0,0), -1)                  
    cv2.fillPoly(image, new_points, colour)
    return image   

if __name__ == '__main__':

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMAGE_DIR)
    if os.path.isdir(base_dir) != True:
        os.mkdir(base_dir)
    
    base_dir = os.path.join(base_dir, SHAPE)
    if os.path.isdir(base_dir) != True:
        os.mkdir(base_dir)

    dir = os.path.join(base_dir, TRAIN_DIR)  #Stepped angles
    print("Generating training images files in {0}".format(dir))
    generate_image_batch(dir, int(NUM_IMAGES * 0.5), False)

    dir = os.path.join(base_dir, VALIDATE_DIR) #Stepped angles
    print("Generating validation images files in {0}".format(dir))
    generate_image_batch(dir, int(NUM_IMAGES * 0.25), False)

    dir = os.path.join(base_dir, TEST_DIR) #Continuous angles
    print("Generating test images files in {0}".format(dir))
    generate_image_batch(dir, int(NUM_IMAGES * 0.25), True)

    
        
