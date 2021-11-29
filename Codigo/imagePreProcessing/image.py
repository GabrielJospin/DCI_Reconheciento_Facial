class image:

    def __init__(self, name, test=False):

        URLTest = "../DataBase/test/original/"
        URLTraining = "../DataBase/training/original/"

        self.name = name
        if(test):
            self.url = URLTest + name + ".jpg"
        else:
            self.url = URLTraining + name + ".jpg"


    def getUrl(self):
        return self.url

    def getName(self):
        return self.name