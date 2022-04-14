import numpy
from skimage.util import random_noise
from skimage.transform import rotate

def normalize_pixels(pixel_matrix : numpy.ndarray) -> numpy.ndarray:
    """Divide all the values inside the given ndarray by 255 to normalize pixels value."""
    
    return pixel_matrix / 255

def get_cnn_prepared_features(features : numpy.ndarray) -> numpy.ndarray:
    """Reshape and normalize the given features array to be usable for CNN."""
    
    # 1: normalize values --> Transform the pixels value from (0,255) into (0,1) so that it can be used by CNN
    
    # 2: CNN needs a vector reprensting the possible channels for one pixel so we have to reshape the data
    # We add just one channel as we stay with greyscale images
    
    return normalize_pixels(features).reshape(-1, 28, 28, 1)

def get_rotated_images(images : numpy.ndarray) -> numpy.ndarray :
    return numpy.array([rotate(image, angle=90) for image in images])

def get_noisy_images(images : numpy.ndarray) -> numpy.ndarray :
    return numpy.array([random_noise(image, mode='salt', amount=0.001) for image in images])