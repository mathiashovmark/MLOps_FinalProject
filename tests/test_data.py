from tests import _PATH_DATA
_PATH_DATA
from data import mnist
import numpy as np
import os
import pytest

train, test = mnist()
training = train['images']
testing = test['images']
len(training)

n_train=5000
n_test=5000

train['labels']

assert len(training) == n_train
assert len(testing) == n_test
for i in range(len(testing)):
    assert testing[i].shape == (28, 28)
    
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def labelTest():
    for i in range(10):
        assert i in np.unique(train['labels']), "Dataset did not have the correct number of targets"

a=10
b="7"
@pytest.mark.parametrize("test_input, expected", [(b, a), ("2+4", 6), ("6*9", 42)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected
