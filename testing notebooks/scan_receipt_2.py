import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.filters import threshold_local
from PIL import Image

# Sample file out of the dataset
file_name = 'receipt_example.jpeg'
img = Image.open(file_name)
img.thumbnail((800,800), Image.ANTIALIAS)
img