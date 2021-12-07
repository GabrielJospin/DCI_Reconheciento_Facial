import cv2
from .constants import *
from skimage import feature


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

    def generate(self, eps=1e-7):
        print(f"start to {self.original.getUrl()}")
        image = cv2.imread(self.originalUrl)
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp_result = feature.local_binary_pattern(imageGray, self.numPoints, self.radius, method="uniform")
        cv2.imwrite(self.finalUrl, lbp_result*255)
        print(f"finish to {self.finalUrl}")