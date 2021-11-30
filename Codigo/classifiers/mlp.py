import numpy as np
import pandas as pd

class mlp:

    def __init__(self, X, Y, h=1):
        self.X = pd.DataFrame(X)
        self.Y = pd.DataFrame(Y)
        self.h = h
        self.ne = len(self.X)
        self.n = len(self.X.columns)

        print(f'mlp of a dataframe {self.ne}X{self.n} with {self.h}  neurons occult')

