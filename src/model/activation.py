import numpy as np


def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return sig


if __name__ == "__main__":
    z = 0
    print(sigmoid(z))