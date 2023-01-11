import torch
import numpy as np

def mnist():
    
    # exchange with the corrupted mnist dataset
    train = np.load("data/train_0.npz")
    test = np.load("data/test.npz")

    return train, test