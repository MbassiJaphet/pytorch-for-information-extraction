# resolve imports for orientation module
import os
import cv2
import glob
import time
import torch
import utils
import torchvision
import numpy as np

#%matplotlib inline 
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

# define transforms to convert PIL image to torch.Tensor
imgToTensor = torchvision.transforms.ToTensor()
# define transforms to convert torch.Tensor to PIL image
tensorToPIL = torchvision.transforms.ToPILImage()