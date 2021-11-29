import constants

class image:

    def __init__(self, name, test=False):

        URLTest = constants.ORIGINAL_TEST_PATH
        URLTraining = constants.ORIGINAL_TRAINING_PATH

        self.name = name
        if(test):
            self.url = URLTest + name + constants.FORMAT
        else:
            self.url = URLTraining + name + constants.FORMAT


    def getUrl(self):
        return self.url

    def getName(self):
        return self.name