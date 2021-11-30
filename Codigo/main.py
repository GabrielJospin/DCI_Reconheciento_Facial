import imagePreProcessing as ipp
import classifiers as clf
from Codigo import constants

from Codigo.readFile import readFile


def print_hi(name):
    print(f'Hello there \nGeneral {name}!!')


def imageArray():
    image = []
    i = 0
    for file_path in constants.file_path:
        rf = readFile(file_path, (i % 2 == 0))
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
        hog.generate()


def preProcessLbp():
    image = imageArray()

    for img in image:
        imageF = ipp.image(img[0], test=(img[1] > 1))
        hog = ipp.lbp(imageF, test=(img[1] > 1))
        hog.generate()


if __name__ == '__main__':
    print_hi('Kenobi')

# preProcessHog()
# preProcessLbp()
df = clf.dataFrame(constants.file_path[0], True, 'hog')
df2 = clf.dataFrame(constants.file_path[1], False, 'hog')

df.generateFiles()
df.generateDB()

X = df.X.append(df2.X, ignore_index=True)
Y = df.Y.append(df2.Y, ignore_index=True)

mlp = clf.mlp(X, Y)
