from Codigo.constants import *

class image():

    def __init__(self, name, test=False):

        URLTest = ORIGINAL_TEST_PATH
        URLTraining = ORIGINAL_TRAINING_PATH

        self.name = name
        if test:
            self.url = URLTest + name + FORMAT
        else:
            self.url = URLTraining + name + FORMAT

    def getUrl(self):
        return self.url

    def getName(self):
        return self.name
