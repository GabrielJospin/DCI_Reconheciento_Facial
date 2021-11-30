from .constants import *
from skimage import feature
import numpy as np
import cv2


class lbp:

    def __init__(self, image, test=False, radius=1, numPoints=1):

        if test:
            lbpURL = LBP_TEST_PATH
        else:
            lbpURL = LBP_TRAINING_PATH

        self.original = image
        self.originalUrl = image.getUrl()
        self.nameImage = image.getName()
        self.finalUrl = lbpURL + self.nameImage + FORMAT
        self.radius = radius
        self.numPoints = numPoints

    def gerate(self):
        print(f"start to {self.original.getUrl()}")
        image = cv2.imread(self.originalUrl)
        cv2.imshow("Original", image)
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray', imageGray)
        lbp_result = feature.local_binary_pattern(imageGray, self.numPoints, self.radius, method="uniform")
        cv2.imwrite(self.finalUrl, lbp_result)
        cv2.imshow('LBP', lbp_result)
        cv2.waitKey(0)
        print(f"finish to {self.finalUrl}")