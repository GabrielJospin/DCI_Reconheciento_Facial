import numpy as np
import pandas as pd


class mlp:

    def __init__(self, X, Y, h=1, itr=20, learn=0.1):
        # Definido Entradas e saidas
        self.X = pd.DataFrame(X)
        self.Y = np.asarray(Y)
        self.h = h
        self.ne = len(self.X)
        self.n = len(self.X.columns)
        self.ns = len(self.Y)
        print(f'mlp of a dataframe {self.ne}X{self.n} with {self.h}  neurons occult \nexit: {self.ns}')

        # Gradientes
        self.w0 = np.random.random((self.n, self.h))
        self.w1 = np.random.random((self.h, self.n))
        self.w2 = np.random.random((self.n, self.h))

        # Outros
        # self.sigmoide = np.vectorize(self.sigmoid(self.X))
        self.itr = itr
        self.learn = learn
        ones = np.ones((4400, 1))
        self.error = ones.__add__(self.Y * -1)

    def train(self):
        print(f'training')
        nep = 0
        E = np.matmul(self.error.transpose(), self.error)
        print(E)
        print(f'errooooo')
        self.feed(self.X)
        g = np.matmul(self.z0, np.matmul(self.z1, self.z2))
        norm = np.linalg.norm(g)
        print(f'pre while')
        print(g)
        print(norm)
        print(nep)
        print(E)
        while norm > 1e-3 and nep < self.itr and E >1e-4:
            print(f'iteration: {nep}')
            nep += 1
            self.w0 -= self.learn * np.matmul(np.matmul(self.z0, g).transpose(), self.z1)
            self.w1 -= self.learn * np.matmul(self.z1.transpose(), g)
            self.w2 -= self.learn * self.z2.transpose()
            self.feed(self.X)
            g = np.matmul(self.z0, np.matmul(self.z1, self.z2))
            g = np.asarray(g)
            norm = np.linalg.norm(g)
            Yr = self.output2
            self.erro = Yr - self.Y
            E = np.matmul(self.error.transpose(), self.error)


    def sigmoid(self, X):
        X = X.astype(float)
        return (1 + np.exp(-X)) ** (-1)

    def answer(self, X):
        print('calc answer')

        Zin = np.matmul(X, self.w0)
        Z = self.sigmoid(Zin)
        Win = np.matmul(Z, self.w1)
        W = self.sigmoid(Win)
        Yin = np.matmul(W, self.w2)
        Y = self.sigmoid(Yin)
        return Y

    def feed(self, X):
        print('calc gradient')
        Zin = np.matmul(X, self.w0)
        Z = self.sigmoid(Zin)
        Win = np.matmul(Z, self.w1)
        W = self.sigmoid(Win)
        Yin = np.matmul(W, self.w2)
        Y = self.sigmoid(Yin)
        self.output2 = Y
        erro = self.output2 - self.Y
        gl = np.matmul((- Y + 1).transpose(), self.Y)
        # print(f'gl:')
        # print(gl)
        # todo: corrigir zz
        Eg = np.matmul(erro, gl)
        # print(Eg)
        self.z0 = np.matmul(Z, np.asarray(Eg).transpose())
        self.z1 = np.matmul(np.aszarray(self.z0), B)
        self.z2 = np.matmul(np.asarray(self.z1).transpose(), X)
