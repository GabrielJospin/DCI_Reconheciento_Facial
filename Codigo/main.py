import readFile
import numpy as np
import pandas as pd
import constants as con
import classifiers as clf
import imagePreProcessing as ipp
from Codigo.classifiers import cnn
from classifiers import constants


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


def preProcessCNN():
    image = imageArray()

    for img in image:
        imageF = ipp.image(img[0], test=(img[1] > 1))
        cnn = ipp.CNNpp(imageF, test=(img[1] > 1))
        hist = cnn.generate()


def gerateDF(data):
    data.generateFiles()
    data.generateDB()


def oper(ValuesIn, ValuesOut, test, ope, modelo):
    classif = ope(ValuesIn, ValuesOut)
    classif.train()
    w = pd.DataFrame(classif.wih)
    print(ValuesIn)
    print(test)
    w.to_csv(f'{con.EXECUCAO_PATH}/exit/{modelo}.{ope.__name__}.data.csv')
    outY = pd.DataFrame(classif.calc_saida(test))
    return outY


if __name__ == '__main__':
    print_hi('Kenobi')

print(f'---------------------------------CNN----------------------------------------')
# preProcessCNN()
print(f'----------------------------------HOG---------------------------------------')
# preProcessHog()
print(f'------------------------------------LBP-------------------------------------')
# preProcessLbp()
print(f'-------------------------------------------------------------------------')
modelos = ['lbp', 'hog']
# for modelo in modelos:
#
#     print(f"---------------- modelo:{modelo} --------------------------- ")
#
#     df = clf.dataFrame(constants.file_path[0], True, modelo)
#     df2 = clf.dataFrame(constants.file_path[1], False, modelo)
#
#     dfTest = clf.dataFrame(constants.file_path[2], True, modelo, test=True)
#     df2Test = clf.dataFrame(constants.file_path[3], False, modelo, test=True)
#
#     gerateDF(df)
#     gerateDF(df2)
#     gerateDF(dfTest)
#     gerateDF(df2Test)
#
#     (X1, Y1) = df.getDB()
#     (X2, Y2) = df2.getDB()
#     (Xt1, Yt1) = dfTest.getDB()
#     (Xt2, Yt2) = df2Test.getDB()
#
#     X = X1.append(X2, ignore_index=True)
#     Y = Y1.append(Y2, ignore_index=True)
#
#     Xt = Xt1.append(Xt2, ignore_index=True)
#     Yt = Yt1.append(Yt2, ignore_index=True)
#     Yt = np.asarray(Yt)
#
#     # print(X)
#     # print(Y)
#     # print(Xt)
#     # print(Yt)
#
#     newX = pd.DataFrame(X)
#     newXt = pd.DataFrame(Xt)
#     newY = pd.DataFrame(Y)
#     newYt = np.asarray(Yt)
#
#     classificadores = [clf.mlp, clf.svm]
#
#     for ope in classificadores:
#
#         print(f"---------------- class:{ope.__name__} --------------------------- ")
#         erro = 0
#         outs = []
#         Y = pd.DataFrame(oper(newX, newY, newXt, ope, modelo))
#
#         for index, x in Y.iterrows():
#             if x[0] > 0.5:
#                 out = 1
#             else:
#                 out = 0
#
#             outs.append(out)
#             if newYt[index] != out:
#                 erro += 1
#
#         error = pd.DataFrame(Y.__sub__(newYt))
#         error.to_csv(f'{con.EXECUCAO_PATH}/exit/{modelo}.{ope.__name__}.error.csv')
#
#         print(np.power(error, 2).sum() / len(error))
#         print((1 - erro / len(newYt)) * 100)

df = clf.dataFrame(constants.file_path[0], True, 'cnn')
df2 = clf.dataFrame(constants.file_path[1], False, 'cnn')
dfTest = clf.dataFrame(constants.file_path[2], True, 'cnn', test=True)
df2Test = clf.dataFrame(constants.file_path[3], False, 'cnn', test=True)

# gerateDF(df)
# gerateDF(df2)
# gerateDF(dfTest)
# gerateDF(df2Test)

(X1, Y1) = df.getDB()
(X2, Y2) = df2.getDB()
(Xt1, Yt1) = dfTest.getDB()
(Xt2, Yt2) = df2Test.getDB()

X = X1.append(X2, ignore_index=True)
Y = Y1.append(Y2, ignore_index=True)

Xt = Xt1.append(Xt2, ignore_index=True)
Yt = Yt1.append(Yt2, ignore_index=True)
Yt = np.asarray(Yt)

newX = np.asarray(X)
newXt = np.asarray(Xt)
newY = np.asarray(Y)
newYt = np.asarray(Yt)

cnn = cnn(newX, newY)
cnn.train()
cnn.calc_saida(newXt, newYt)

# Y = pd.DataFrame(oper(newX, newY, newXt, cnn, 'cnn'))

print(f'Hasta La Vista, baby')
