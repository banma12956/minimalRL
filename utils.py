import numpy as np

class normalization():
    def __init__(self, shape, demean=True):
        self.shape = shape
        self.mean = np.zeros(self.shape)
        self.var = np.zeros(self.shape)
        self.n = 0
        self.demean = demean

    def push(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 0:
            self.mean[...] = x
        else:
            old_mean = self.mean.copy()
            self.mean[...] = old_mean + (x - old_mean) / self.n
            self.var[...] = self.var + (x - old_mean) * (x - self.mean)

    def __call__(self, x, p=True):
        if p:
            self.push(x)
        if self.demean:
            x = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        else:
            x = x / (np.sqrt(self.var) + 1e-8)
        return x
