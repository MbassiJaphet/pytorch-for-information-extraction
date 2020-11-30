# resolve imports for orientation module
import os
import cv2
import time
import torch
import torchvision

import numpy as np

import utils

from PIL import Image, ImageDraw
#%matplotlib inline 
from matplotlib import pyplot as plt

# define transforms to convert PIL image to torch.Tensor
imgToTensor = torchvision.transforms.ToTensor()
# define transforms to convert torch.Tensor to PIL image
tensorToPIL = torchvision.transforms.ToPILImage()