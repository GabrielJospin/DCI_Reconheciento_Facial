
import cv2
import numpy as np
import skimage
from skimage import filters, img_as_float
import scipy.ndimage as ndi


from .constants import *


class CNNpp:

    def __init__(self, image, test=False):

        if test:
            CnnUrl = CNN_TEST_PATH
        else:
            CnnUrl = CNN_TRAINING_PATH

        self.original = image
        self.originalUrl = self.original.getUrl()
        self.nameImage = self.original.getName()
        self.finalUrl = CnnUrl + self.nameImage + FORMAT

    def generate(self, eps=1e-7):
        print(f"start to {self.original.getUrl()}")
        image = cv2.imread(self.originalUrl)
        final = filters.laplace(image, ksize=3)
        cv2.imwrite(self.finalUrl, final*255)
        print(f"finish to {self.finalUrl}")
