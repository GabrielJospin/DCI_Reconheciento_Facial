
import numpy as np


class mlp():

    @staticmethod
    def normalize(X: np.array):
        tam = len(X)
        Y = (X - X.mean()) / X.std()
        return Y

    def __init__(self, X, Y, h=1):
        super(mlp, self).__init__()
        self.X = self.normalize(X)
        self.Y = Y
        self.h = h
        N, ne = X.shape
        ns = Y.shape[1]
        self.A = np.ones((self.h, ne))
        self.A = 2 * self.A - 1
        self.B = np.ones((ns, self.h))
        self.B = 2 * self.B - 1

    def train(self):
        h = self.h
        X = self.X
        Y = self.Y
        N, ne = X.shape
        ns = Y.shape[2 - 1]
        # Inicializei a matriz de pesos
        A = self.A
        B = self.B
        # Calcula do erro
        Yr = self.calc_saida(X)
        print(Y)
        print(Yr)
        erro = np.matrix(np.subtract(Y, Yr))
        print(erro)
        E = sum(sum(np.multiply(erro, erro)))
        # Definir numero de epocas
        nepmax = 200
        nep = 0
        alfa = 0.01
        # Calculo do gradiente
        self.calc_grad()
        g = np.concatenate([self.dEdA.transpose(), self.dEdB])
        print(np.linalg.norm(g))
        print(E)
        print('--------------------------------------------')
        while np.linalg.norm(g) > 0.001 and (nep < nepmax and E > 0.0001):
            print(f'iteration {nep}')
            # Incrementar o numero de epocas
            nep = nep + 1
            # Atualiza os pesos
            A = A - alfa * self.dEdA
            B = B - alfa * self.dEdB
            # Calculo o gradiente
            self.calc_grad()
            g = np.concatenate([self.dEdA.transpose(), self.dEdB])
            # Calculo o erro
            Yr = self.calc_saida(X)
            erro = np.matrix(np.subtract(Y, Yr))
            E = sum(sum(np.multiply(erro, erro)))
            self.A = A
            self.B = B
            print(np.linalg.norm(g))
            print(E)
            print('--------------------------------------------')

    def calc_saida(self, X):
        X = self.normalize(X)
        A = self.A
        B = self.B
        N, ne = X.shape
        Zin = np.matmul(X, np.transpose(A))
        Z = 1.0 / (1 + np.exp(Zin))
        Yin = np.matmul(Z, np.transpose(B))
        Yr = 1.0 / (1 + np.exp(Yin))
        return Yr * 2

    def calc_grad(self):
        A = self.A
        B = self.B
        X = self.X
        Y = self.Y

        Zin = np.matmul(X, np.transpose(A))
        Z = 1.0 / (1 + np.exp(Zin))
        Yin = np.matmul(Z, np.transpose(B))
        Yr = 1.0 / (1 + np.exp(Yin))
        erro = np.subtract(Yr, Y)
        gl = np.multiply((1 - Yr), Yr)
        fl = np.multiply((1 - Z), Z)
        self.dEdB = np.matmul(np.multiply(erro, gl).transpose(), Z)
        self.dEdZ = np.matmul(np.multiply(erro, gl), B)
        self.dEdA = np.matmul(np.transpose((np.multiply(self.dEdZ, fl))), X)
