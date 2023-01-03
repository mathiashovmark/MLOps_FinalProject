import torch
import numpy as np


def mnist():
    # exchange with the corrupted mnist dataset
    train = np.load("C:/Users/mathi/Desktop/dtu_mlops/data/corruptmnist/train_0.npz")
    #train = torch.from_numpy(train)
    test = np.load("C:/Users/mathi/Desktop/dtu_mlops/data/corruptmnist/test.npz")
    #test= torch.from_numpy(test)
    return train, test