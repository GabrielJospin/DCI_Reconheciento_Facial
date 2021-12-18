import numpy as np
import pandas as pd
from .constants import *
import cv2


class dataFrame:

    def __init__(self, pathFile, pairs, model, test=False):

        if model == 'hog' and test:
            self.pathDic = HOG_TEST_PATH
        elif model == 'hog' and not test:
            self.pathDic = HOG_TRAINING_PATH
        elif model == 'lbp' and test:
            self.pathDic = LBP_TEST_PATH
        elif model == 'lbp' and not test:
            self.pathDic = LBP_TRAINING_PATH
        else:
            raise ValueError("model is not acept")
        self.model = model
        if pairs:
            self.type = 'pairs'
        else:
            self.type = 'notPairs'
        self.path = pathFile
        self.pairs = pairs
        self.files = pd.DataFrame(columns=['img1', 'img2', 'pair'])
        self.X = pd.DataFrame()
        self.Y = pd.DataFrame(columns=['pair'])
        self.db = pd.DataFrame()


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
            max_bins = int(image1.max() + 1)
            (hist1, _) = np.histogram(image1.ravel(), normed=True, bins=max_bins, range=(0, max_bins))

            image2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2GRAY)
            (hist2, _) = np.histogram(image2.ravel(), normed=True, bins=max_bins, range=(0, max_bins))

            x = np.concatenate([hist1, hist2])
            if len(self.X) == 0:
                self.X = pd.DataFrame(x).transpose()
            else:
                zipx = zip(self.X.columns, x)
                dic = dict(zipx)
                self.X = self.X.append(dic, ignore_index=True)
            self.Y = self.Y.append({'pair': y}, ignore_index=True)

        self.db = self.X
        self.db['pair'] = self.Y['pair']
        self.db.to_csv(f'{self.pathDic}{self.model}.{self.type}.csv')

    def getDB(self):
        df = pd.read_csv(f'{self.pathDic}{self.model}.{self.type}.csv', index_col=0)
        X = df.iloc[:, 0:(len(df.columns) - 1)]
        Y = df.iloc[:,(len(df.columns) - 1):(len(df.columns))]
        return (X, Y)
