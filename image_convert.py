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

CONVERT_TYPE =  cv2.COLOR_BGR2GRAY
base_dir = 'G:\Team Drives\R&D\Development\Classified Image Test'
output_dir = 'C:\Test\GreyscaleResized'
SIZE = (254, 254)

if __name__ == '__main__':    
    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir() ]
    #print(subfolders)
    for folder in subfolders:
        foldername = folder.split('\\')[-1]
        print("Examining folder {0}....".format(foldername))
        file_names = os.listdir(folder)
        print("Number of files found = {0}.".format(len(file_names)))
        png_list = [i for i in file_names if ".png" in i]
        print("Number of png files found = {0}".format(len(png_list)))
        for input_file in png_list:
            output_filename = os.path.join(output_dir, foldername, input_file.split('\\')[-1])
            print("Converting {0}...".format(input_file))
            #load
            file_to_load = os.path.join(base_dir, foldername, input_file)
            im = cv2.imread(file_to_load)
            #greyscale
            grey_im = cv2.cvtColor(im, CONVERT_TYPE)
            #resize
            smaller = cv2.resize(grey_im, SIZE)
            #save
            #make directory if not already there already
            dir = os.path.join(output_dir,foldername)
            if os.path.isdir(dir) != True:
                os.mkdir(dir)
            #print(output_filename)
            cv2.imwrite(output_filename, smaller)
            print("Done.")

