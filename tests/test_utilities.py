import numpy as np
import sys
sys.path.append("../behavioral_analysis")
import utilities
import pytest
from hypothesis import given
import hypothesis.strategies
import hypothesis.extra.numpy

def test_fix_FLIR_file_order_array_shape():
    arr = ((1, 2), (1, 2))
    with pytest.raises(RuntimeError) as excinfo:
        utilities.fix_FLIR_file_order(arr)
    excinfo.match('The im_files object must be 1D, provided object has shape \(2, 2\)')
    
def test_fix_FLIR_file_order_array_like():
    arr = '((1, 2), (1, 2))'
    with pytest.raises(RuntimeError) as excinfo:
        utilities.fix_FLIR_file_order(arr)
    excinfo.match('im_files must be array like, provided object has type <class \'str\'>')