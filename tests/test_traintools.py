import numpy as np
import sys
sys.path.append("../behavioral_analysis")
import traintools
import pytest
from hypothesis import given
import hypothesis.strategies
import hypothesis.extra.numpy

def test_im_shape():
    im = np.array([
        [[[1, 2], [1, 2]], [[1, 2], [1, 2]]],
        [[[1, 2], [1, 2]], [[1, 2], [1, 2]]]
        ])
    with pytest.raises(RuntimeError) as excinfo:
        traintools._check_ims(im)
    excinfo.match("Need to provide an array with shape \(n, m\). Provided array has shape \(2, 2, 2\)")
    
def test_ims_shape():
    ims = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    traintools._check_ims(ims)
    
def test_ims_data_type_list():
    ims = [[[1, 2], [1, 2]], [[1, 2], [1, 2]]]
    with pytest.raises(RuntimeError) as excinfo:
        traintools._check_ims(ims)
    excinfo.match("Need to provide a numpy array, image has type <class 'list'>")
    
def test_ims_data_type_contrains_string():
    ims = np.array([[['1', 2], [1, 2]], [[1, 2], [1, 2]]])
    with pytest.raises(RuntimeError) as excinfo:
        traintools._check_ims(ims)
    excinfo.match("Provided image has unsuported type: <U1")

def test_im_data_type_string():
    im = '[[[1, 2], [1, 2]], [[1, 2], [1, 2]]]'
    with pytest.raises(RuntimeError) as excinfo:
        traintools._check_ims(im)
    excinfo.match("The given ims object is not array like, it is <class 'str'>")
