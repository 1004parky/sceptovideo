import numpy as np
import sys
sys.path.append("../behavioral_analysis")
import segmentation
import pytest
import skimage.filters
from hypothesis import given
import hypothesis.strategies
import hypothesis.extra.numpy

# test functions for simple segmentation based tracking code

def test_im_shape():
    im = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_image_input(im)
    excinfo.match("Need to provide an array with shape \(n, m\). Provided array has shape \(2, 2, 2\)")
    
def test_im_data_type_list():
    im = [[1, 2, 3], [1, 2, 3]]
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_image_input(im)
    excinfo.match("Need to provide a numpy array, image has type <class 'list'>")
    
def test_im_data_type_string():
    im = '[[1, 2, 3], [1, 2, 3]]'
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_image_input(im)
    excinfo.match("Need to provide a numpy array, image has type <class 'str'>")
    
def test_im_shape_segment():
    im = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im)
    excinfo.match("Need to provide an array with shape \(n, m\). Provided array has shape \(2, 2, 2\)")
    
def test_im_data_type_list_segment():
    im = [[1, 2, 3], [1, 2, 3]]
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im)
    excinfo.match("Need to provide a numpy array, image has type <class 'list'>")
    
def test_im_data_type_string_segment():
    im = '[[1, 2, 3], [1, 2, 3]]'
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im)
    excinfo.match("Need to provide a numpy array, image has type <class 'str'>")
    
def test_provided_function_callable():
    im = np.array([[1, 2, 3], [1, 2, 3]])
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im, thresh_func='Hello, world.')
    excinfo.match("The provided function is not callable")
    
def test_provided_function_callable_mat():
    im = np.array([[1, 2, 3], [1, 2, 3]])
    args = (3,)
    assert segmentation._check_function_input(im, skimage.filters.threshold_local, args) == True

def test_provided_function_returns_correct_shape():
    im = np.array([[1, 2, 3], [1, 2, 3]])
    def bad_func(im):
        return(np.array([[1, 2], [1, 2]]))
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im, thresh_func=bad_func)
    excinfo.match("Array output of the function must have same shape as the image \
                           the output array has shape \(2, 2\), image has shape \(2, 3\)")

def test_provided_function_returns_correct_types():
    im = np.array([[1, 2, 3], [1, 2, 3]])
    def bad_func(im):
        return('Hello, world!')
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im, thresh_func=bad_func)
    excinfo.match("The provided function must output a numeric or array \
                           provided function returns type <class 'str'>")
    
def test_check_numeric_function():
    assert segmentation._check_numeric_types(np.int32(1)) == True
    
def test_bg_subtract_im_type():
    im1 = np.array([[1, 2, 3], [1, 2, 3]])
    im2 = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.bg_subtract(im1, im2)
    excinfo.match("Need to provide an array with shape \(n, m\). Provided array has shape \(2, 2, 2\)")
    
def test_bg_subtract_im_dims():
    im1 = np.array([[1, 2, 3], [1, 2, 3]])
    im2 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.bg_subtract(im1, im2)
    excinfo.match("The provided images have different dimension \
        im1: \(2, 3\), im2: \(3, 3\)")
    
def test_im_normalization_range():
    im = np.array([[1, 2, 3], [1, 2, 3]])
    new_im = segmentation.normalize_convert_im(im)
    assert new_im.max() == 1
    assert new_im.min() == 0
   
@given(hypothesis.extra.numpy.arrays(dtype=int, shape=(50,50)))
def test_im_normalization_range_int(im):
    
    if np.isclose(im.max(), 0) and np.isclose(im.min(), 0):
        with pytest.raises(RuntimeError) as excinfo:
            segmentation.normalize_convert_im(im)
        excinfo.match("Inputed image is near to zero for all values")
    elif np.isclose((im.max() - im.min()), 0):
        with pytest.raises(RuntimeError) as excinfo:
            segmentation.normalize_convert_im(im)
        excinfo.match("Inputed image has nearly the same value for all pixels. Check input")
    else:
        new_im = segmentation.normalize_convert_im(im)
        assert new_im.max() == 1
        assert new_im.min() == 0
        
@given(hypothesis.extra.numpy.arrays(dtype=float, shape=(50,50)))
def test_im_normalization_range_float(im):
    
    if np.isclose(im.max(), 0) and np.isclose(im.min(), 0):
        with pytest.raises(RuntimeError) as excinfo:
            segmentation.normalize_convert_im(im)
        excinfo.match("Inputed image is near to zero for all values")
    elif np.any(np.isnan(im)):
        with pytest.raises(RuntimeError) as excinfo:
            segmentation.normalize_convert_im(im)
        excinfo.match("Data contains a nan, decide how to handle missing data")
    elif np.any(np.isinf(im)):
        with pytest.raises(RuntimeError) as excinfo:
            segmentation.normalize_convert_im(im)
        excinfo.match("Data contains an np.inf, decide how to handle infinite values")
    elif np.isclose((im.max() - im.min()), 0):
        with pytest.raises(RuntimeError) as excinfo:
            segmentation.normalize_convert_im(im)
        excinfo.match("Inputed image has nearly the same value for all pixels. Check input")
    else:
        new_im = segmentation.normalize_convert_im(im)
        assert new_im.max() == 1
        assert new_im.min() == 0
        
@given(hypothesis.extra.numpy.arrays(dtype=np.float128, shape=(50,50)))
def test_im_normalization_range_float128(im):
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.normalize_convert_im(im)
    excinfo.match("Provided image has unsuported type: float128")

def test_im_near_zero():
    im = np.array([[0, 0, 0], [0, 0, 0]])
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im)
    excinfo.match("Inputed image is near to zero for all values")

def test_im_has_nan():
    im = np.array([[np.nan, 0, 0], [0, 0, 0]])
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im)
    excinfo.match("Data contains a nan, decide how to handle missing data")

def test_im_has_nan():
    im = np.array([[np.inf, 0, 0], [0, 0, 0]])
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.segment(im)
    excinfo.match("Data contains an np.inf, decide how to handle infinite values")
