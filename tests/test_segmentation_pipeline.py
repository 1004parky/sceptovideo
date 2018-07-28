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
       
def test_int_types():
    assert segmentation._check_int_types(1.0) == False
    assert segmentation._check_int_types(1) == True
    assert segmentation._check_int_types(np.int32(1)) == True
    assert segmentation._check_int_types(np.uint64(1)) == True
    
@given(hypothesis.extra.numpy.arrays(dtype=np.float128, shape=(10,10)))
def test_numpy_array_int_types(ar):
    segmentation._check_numpy_array_int_types(ar) == False
    
@given(hypothesis.strategies.tuples(hypothesis.strategies.integers(), hypothesis.strategies.integers()))
def test_array_like_tuple(tu):
    segmentation._check_array_like(tu) == True

@given(hypothesis.strategies.lists())
def test_array_like_list(li):
    segmentation._check_array_like(li) == True
    
@given(hypothesis.extra.numpy.arrays(dtype=np.float128, shape=(10,10)))
def test_array_like_nparray(ar):
    segmentation._check_array_like(ar) == True
    
def test_numpy_array_string_types():
    segmentation._check_numpy_array_string_types(np.array(['rectangle', 'ellipse'])) == True

def test_check_roi_inputs():
    
    roi_kind = ['ellipse', 'rectangle']
    cent = [1, 2]
    width = [1, 2]
    height = [1, 2]
    outside_roi = 'max'
    
    with pytest.raises(RuntimeError) as excinfo:
            segmentation._check_roi_inputs('hello', cent, width, height, outside_roi)
    excinfo.match("The given roi kind object is not array like, it is " + str(type('hello')))
    
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_roi_inputs(roi_kind, 1, width, height, outside_roi)               
    excinfo.match("The given roi centers object object is not array like, it is " + str(type(1)))
    
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_roi_inputs(roi_kind, cent, 5.0, height, outside_roi)  
    excinfo.match("The given width object object is not array like, it is " + str(type(5.0)))
        
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_roi_inputs(roi_kind, cent, width, '[1.0, 1.2]', outside_roi)      
    excinfo.match("The given height object object is not array like, it is " + str(type('[1.0, 1.2]')))
    
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_roi_inputs(roi_kind, [1.0, 1.2], width, height, outside_roi)  
    excinfo.match("The cent object must have entries of integer type")
         
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_roi_inputs(roi_kind, cent, ['1', 2], height, outside_roi)  
    excinfo.match("The width object must have entries of integer type")
           
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_roi_inputs(roi_kind, cent, width, [(1, 2), 2], outside_roi)  
    excinfo.match("The height object must have entries of integer type")
        
    with pytest.raises(RuntimeError) as excinfo:
        segmentation._check_roi_inputs([1, 2], cent, width, height, outside_roi)  
    excinfo.match("The roi_kind object must have entries of type str")
    
def test_check_crop_inputs():
    
    cent = [1, 2]
    width = 20
    height = 10
    
    cent = 0
    with pytest.raises(RuntimeError) as excinfo:
            segmentation._check_crop_inputs(cent, width, height)
    excinfo.match("The given cent object is not array like, it is " + str(type(cent)))
    
    cent = [1, 2.0]
    with pytest.raises(RuntimeError) as excinfo:
            segmentation._check_crop_inputs(cent, width, height)
    excinfo.match("The cent object must have entries of integer type")
    
    cent = [1, 2]
    width = 2.0
    with pytest.raises(RuntimeError) as excinfo:
            segmentation._check_crop_inputs(cent, width, height)
    excinfo.match("The width must be integer type")
    
    width = 20
    height = 2.0
    with pytest.raises(RuntimeError) as excinfo:
            segmentation._check_crop_inputs(cent, width, height)
    excinfo.match("The height must be integer type")
    
def test_bg_image_maker_string_err():
    im = np.array([[0,1], [0,1]])
    ims = 'im, im, im'
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.construct_bg_img(ims)
    excinfo.match('Provided ims object is not array like, it is type <class \'str\'>')
    
def test_bg_image_maker_im_err():
    im = np.array(['[0,1]', '[0,1]', '[0,1]'])
    ims = [im, im, im]
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.construct_bg_img(ims)
    excinfo.match('Need to provide an array with shape \(n, m\). Provided array has shape \(3,\)')
    
def test_bg_image_maker_im_err():
    im = np.array([[0,1], [0,1]])
    ims = [im, im, im]
    with pytest.raises(RuntimeError) as excinfo:
        segmentation.construct_bg_img(ims, num_ims=10.0)
    excinfo.match('Please provide an integer for the num_ims parameter. Provided argument has type ' + str(type(10.0)))