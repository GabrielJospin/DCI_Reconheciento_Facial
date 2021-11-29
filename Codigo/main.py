import imagePreProcessing as ipp
import pandas as pd


def print_hi(name):
    print(f'Hello there \nGeneral {name}!!')


if __name__ == '__main__':
    print_hi('Kenobi')

pairs = pd.DataFrame(columns=['Person', 'img'])

file_path = ipp.constants.TRAINING_PATH + "/pairsDevTrain" + ".txt"

with open(file_path, "rb") as infile:
    DBlen = int(infile.readline())
    while True:
        text = infile.readline()

        if not text: break
        (person, img1, img2) = text.split()
        person = person.decode('UTF-8')
        img1 = int(img1)
        img2 = int(img2)
        pairs = pairs.append({'Person': person, 'img': img1}, ignore_index=True)
        pairs = pairs.append({'Person': person, 'img': img2}, ignore_index=True)

print(pairs)
image = []

for pair in pairs.itertuples():
    image.append(f"{pair.Person}_{str(pair.img).zfill(4)}")

print(image)

for img in image:
    imageF = ipp.image(img)
    hog = ipp.hog(imageF)
    hog.generate()
