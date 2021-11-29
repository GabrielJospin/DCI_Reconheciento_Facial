import pandas as pd


class readFile:

    def __init__(self, path, pairs):
        self.DBlen = 0
        self.path = path
        self.pairs = pairs
        self.files = pd.DataFrame(columns=['Person', 'img'])

    def generate(self):
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
                    self.files = self.files.append({'Person': person, 'img': img1}, ignore_index=True)
                    self.files = self.files.append({'Person': person, 'img': img2}, ignore_index=True)
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
                    self.files = self.files.append({'Person': person, 'img': img1}, ignore_index=True)
                    self.files = self.files.append({'Person': person2, 'img': img2}, ignore_index=True)
