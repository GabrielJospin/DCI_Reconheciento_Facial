import numpy as np
import pandas as pd

import imagePreProcessing as ipp
import classifiers as clf
from classifiers import constants

import readFile


def print_hi(name):
    print(f'Hello there \nGeneral {name}!!')


def imageArray():
    image = []
    i = 0
    for file_path in constants.file_path:
        rf = readFile.readFile(file_path, (i % 2 == 0))
        rf.generate()
        print(rf.files)

        for file in rf.files.itertuples():
            image.append([f"{file.Person}_{str(file.img).zfill(4)}", i])

        i += 1

    return image


def preProcessHog():
    image = imageArray()

    for img in image:
        imageF = ipp.image(img[0], test=(img[1] > 1))
        hog = ipp.hog(imageF, test=(img[1] > 1))
        hist = hog.generate()


def preProcessLbp():
    image = imageArray()

    for img in image:
        imageF = ipp.image(img[0], test=(img[1] > 1))
        lbp = ipp.lbp(imageF, test=(img[1] > 1))
        hist = lbp.generate()


def gerateDF(data):
    data.generateFiles()
    data.generateDB()


def oper(ValuesIn, ValuesOut, test, ope):
    classif = ope(ValuesIn, ValuesOut)
    classif.train()
    print(test)
    outY = pd.DataFrame(classif.calc_saida(test))
    return outY


if __name__ == '__main__':
    print_hi('Kenobi')

# preProcessHog()
# preProcessLbp()

df = clf.dataFrame(constants.file_path[0], True, 'hog')
df2 = clf.dataFrame(constants.file_path[1], False, 'hog')

dfTest = clf.dataFrame(constants.file_path[2], True, 'hog', test=True)
df2Test = clf.dataFrame(constants.file_path[3], False, 'hog', test=True)

gerateDF(df)
gerateDF(df2)
gerateDF(dfTest)
gerateDF(df2Test)

(X1, Y1) = df.getDB()
(X2, Y2) = df2.getDB()
(Xt1, Yt1) = dfTest.getDB()
(Xt2, Yt2) = df2Test.getDB()

X = X1.append(X2, ignore_index=True)
Y = Y1.append(Y2, ignore_index=True)

Xt = Xt1.append(Xt2, ignore_index=True)
Yt = Yt1.append(Yt2, ignore_index=True)
Yt = np.asarray(Yt)

print(X)
print(Y)
print(Xt)
print(Yt)

newX = pd.DataFrame(X)
newXt = pd.DataFrame(Xt)
newY = pd.DataFrame(Y)
newYt = np.asarray(Yt)

classificadores = [clf.mlp, clf.svm]

for ope in classificadores:

    erro = 0
    outs = []
    Y = pd.DataFrame(oper(newX, newY, newXt, ope))

    for index, x in Y.iterrows():
        if x[0] > 0.5:
            out = 1
        else:
            out = 0

        outs.append(out)
        if newYt[index] != out:
            erro += 1

    print(outs)
    print(newYt)
    print(Y.transpose())
    print(erro)
    print((1 - erro / len(newYt)) * 100)
