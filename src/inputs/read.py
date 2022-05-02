import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

def get_belgium_traffic_signs():
    """Load the resized belgium traffic signs images in a ndarray"""
    
    x_train, y_train, x_test, y_test = [], [], [], []
    
    # Get train images
    for folder in os.scandir("resources/belgium_traffic_signs/training"):
        if folder.is_dir():
            for file in os.scandir(folder.path):
                x_train.append(imread(file.path))
                y_train.append(int(folder.name))
    
    # Get test images  
    for folder in os.scandir("resources/belgium_traffic_signs/training"):
        if folder.is_dir():
            for file in os.scandir(folder.path):
                x_test.append(imread(file.path))
                y_test.append(int(folder.name))
    
    # Give all images same size (128, 128) 
    x_train = np.array([resize(img, (128, 128)) for img in x_train])
    x_test = np.array([resize(img, (128, 128)) for img in x_test])
                
    return (x_train, np.array(y_train)), (x_test, np.array(y_test))