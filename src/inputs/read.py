import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def get_belgium_traffic_signs():
    """Loads and resizes belgium traffic signs images"""
    
    x_train, y_train, x_test, y_test = [], [], [], []
    
    # Get train images
    for folder in os.scandir("resources/belgium_traffic_signs/training"):
        if folder.is_dir():
            for file in os.scandir(folder.path):
                x_train.append(imread(file.path))
                y_train.append(int(folder.name))
    
    # Get test images  
    for folder in os.scandir("resources/belgium_traffic_signs/testing"):
        if folder.is_dir():
            for file in os.scandir(folder.path):
                x_test.append(imread(file.path))
                y_test.append(int(folder.name))
    
    # Give all images same size (128, 128) 
    x_train = np.array([resize(img, (128, 128)) for img in x_train])
    x_test = np.array([resize(img, (128, 128)) for img in x_test])
                
    return (x_train, np.array(y_train)), (x_test, np.array(y_test))

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