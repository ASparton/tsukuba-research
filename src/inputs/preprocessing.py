import numpy
from skimage.util import random_noise
from PIL import Image

def normalize_pixels(pixel_matrix : numpy.ndarray) -> numpy.ndarray:
    """Divide all the values inside the given ndarray by 255 to normalize pixels value."""
    
    return pixel_matrix / 255

def get_rotated_images(images : numpy.ndarray) -> numpy.ndarray :
    """Rotate the given images by 10 degree to the left."""
    
    return numpy.array([numpy.array(Image.fromarray(image).rotate(10)) for image in images])

def get_noisy_images(images : numpy.ndarray) -> numpy.ndarray :
    """Add a bit of noise to the given images."""
    
    return random_noise(images, mode='gaussian', var=0.001)