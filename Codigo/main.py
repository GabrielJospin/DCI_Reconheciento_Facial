import imagePreProcessing as ipp
import pandas as pd

from Codigo.readFile import readFile


def print_hi(name):
    print(f'Hello there \nGeneral {name}!!')

def preProcessHog():
    image = []
    i = 0
    for file_path in ipp.constants.file_path:
        rf = readFile(file_path, (i % 2 == 0))
        rf.generate()
        print(rf.files)

        for file in rf.files.itertuples():
            image.append([f"{file.Person}_{str(file.img).zfill(4)}", i])

        i += 1

    # print(image)

    for img in image:
        imageF = ipp.image(img[0], test=(img[1] > 1))
        hog = ipp.hog(imageF, test=(img[1] > 1))
        hog.generate()


if __name__ == '__main__':
    print_hi('Kenobi')

# preProcessHog()
