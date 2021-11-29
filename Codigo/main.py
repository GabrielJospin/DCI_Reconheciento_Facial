import imagePreProcessing as ipp
import os


def print_hi(name):
    print(f'Hello there \nGeneral {name}!!')


if __name__ == '__main__':
    print_hi('Kenobi')

image = ipp.image("Bill_Gates_0001")
hog = ipp.hog(image)
hog.generate()
