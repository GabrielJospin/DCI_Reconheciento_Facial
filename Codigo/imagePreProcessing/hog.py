from skimage import feature
import cv2
import matplotlib.pyplot as plt
from .constants import *


class hog:

    def __init__(self, image, test=False):

        if test:
            HogUrl = HOG_TEST_PATH
        else:
            HogUrl = HOG_TRAINING_PATH

        self.original = image
        self.originalUrl = self.original.getUrl()
        self.nameImage = self.original.getName()
        self.finalUrl = HogUrl + self.nameImage + FORMAT

    def generate(self):
        print(f"start to {self.original.getUrl()}")
        image = cv2.imread(self.originalUrl)
        (hogF, hog_image) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True,
                                        transform_sqrt=True)
        cv2.imwrite(self.finalUrl, hog_image*255.)
        print(f"finish to {self.finalUrl}")
