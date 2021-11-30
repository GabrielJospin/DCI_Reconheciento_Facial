import pandas as pd
from Codigo import constants
import cv2


class dataFrame:

    def __init__(self, pathFile, pairs, model, test=False):

        if model == 'hog' and test:
            self.pathDic = constants.HOG_TEST_PATH
        elif model == 'hog' and not test:
            self.pathDic = constants.HOG_TRAINING_PATH
        elif model == 'lpb' and test:
            self.pathDic = constants.LBP_TEST_PATH
        elif model == 'lpb' and not test:
            self.pathDic = constants.LBP_TRAINING_PATH
        else:
            raise ValueError("model is not acept")

        self.path = pathFile
        self.pairs = pairs
        self.files = pd.DataFrame(columns=['img1', 'img2', 'pair'])
        self.X = pd.DataFrame(columns=['img1', 'img2'])
        self.Y = pd.DataFrame(columns=['pair'])

    def generateFiles(self):
        if self.pairs:
            with open(self.path, "rb") as infile:
                self.DBLen = int(infile.readline())
                while True:
                    text = infile.readline()

                    if not text: break
                    (person, img1, img2) = text.split()
                    person = person.decode('UTF-8')
                    img1 = int(img1)
                    img2 = int(img2)
                    self.files = self.files.append({'img1': f"{self.pathDic}{person}_{str(img1).zfill(4)}.jpg",
                                                    'img2': f"{self.pathDic}{person}_{str(img2).zfill(4)}.jpg",
                                                    'pair': 1}
                                                   , ignore_index=True)
        else:
            with open(self.path, "rb") as infile:
                self.DBLen = int(infile.readline())
                while True:
                    text = infile.readline()

                    if not text: break
                    (person, img1, person2, img2) = text.split()
                    person = person.decode('UTF-8')
                    person2 = person2.decode('UTF-8')
                    img1 = int(img1)
                    img2 = int(img2)
                    self.files = self.files.append({'img1': f"{self.pathDic}{person}_{str(img1).zfill(4)}.jpg",
                                                    'img2': f"{self.pathDic}{person2}_{str(img2).zfill(4)}.jpg",
                                                    'pair': 0}
                                                   , ignore_index=True)

    def generateDB(self):

        self.generateFiles()
        for index, row in self.files.iterrows():
            img1 = row['img1']
            img2 = row['img2']
            y = int(row['pair'])
            image1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2GRAY)
            image2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2GRAY)
            self.X = self.X.append({'img1': image1, 'img2': image2}, ignore_index=True)
            self.Y = self.Y.append({'pair': y}, ignore_index=True)
