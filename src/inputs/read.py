import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def get_arabic_traffic_signs(nb_images_to_load = -1):
    """Loads and resizes the arabic traffic signs images"""
    
    x_train, y_train, x_test, y_test = [], [], [], []
    
    # Get train images
    for file in os.scandir("resources/arabic_traffic_signs/training"):
        x_train.append(imread(file.path))
        y_train.append(int(file.name.split("_")[0]))
        
        if nb_images_to_load == 0:
            break
        nb_images_to_load -= 1
    
    # Get test images
    for file in os.scandir("resources/arabic_traffic_signs/testing"):
        x_test.append(imread(file.path))
        y_test.append(int(file.name.split("_")[0]))
        
        if nb_images_to_load == 0:
            break
        nb_images_to_load -= 1
    
    # Give all images same size (512, 512) 
    x_train = np.array([resize(img, (512, 512)) for img in x_train])
    x_test = np.array([resize(img, (512, 512)) for img in x_test])
                
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))