from math import exp, sqrt

import numpy as np


class RegLog:

    @staticmethod
    def normalize(X: np.array):
        tam = len(X)
        Y = (X - X.mean()) / X.std()
        return Y

    def __normalize__(self):
        return (self.X - self.X.mean()) / self.X.std()

    def predict(self, X):
        X = self.normalize(X)
        result = []
        for (i, j), x in np.ndenumerate(X):
            if j == 0:
                poli = self.b1.__matmul__(np.matrix(X[i]))
                w = -self.b0 - np.sum(poli)
                factor = 1 / (1 + np.exp(w))
                result.append(factor)
        return result

    def result(self, matrizDados):
        brutal = self.predict(matrizDados)
        for index, ele in np.ndenumerate(np.array(brutal)):
            if ele >= 0.5:
                brutal[index[0]] = 1
            else:
                brutal[index[0]] = 0

        return brutal

    def __init__(self, X, Y):
        self.X = np.matrix(X)
        self.X = self.__normalize__()
        self.Y = Y
        self.b0 = 1
        self.b1 = np.ones((1, 20)).transpose()
        L = 0.001
        epochs = int(len(self.X)/20)
        D_b0 = 0
        D_b1 = 0

        for epoch in range(epochs):
            print(f'iteration{epoch} of {epochs}')
            y_pred = np.matrix(self.predict(self.X)).transpose()
            temp_Y = np.array(self.Y)
            temp_X = np.array(self.X)
            D_b0 = -2 * sum((temp_Y - y_pred) * y_pred.transpose() * (1 - y_pred))
            D_b1 = -2 * ((temp_X.transpose() * (temp_Y - y_pred)) * (y_pred.transpose() * (1 - y_pred)))

            # print(f'derivade')
            # print(f'D_b0:\n{D_b0}')
            # print(f'D_b1:\n{D_b1}')

            self.b0 = self.b0 - L * D_b0[0, 0]
            self.b1 = np.matrix(self.b1 - L * D_b1)
            # print(f'coef')
            # print(f'b0:\n{self.b0}')
            # print(f'b1:\n{self.b1}')
