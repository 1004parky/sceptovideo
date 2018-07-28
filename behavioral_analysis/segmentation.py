import numpy as np
import skimage.io
import skimage.filters
import pandas as pd

def _check_image_input(im):
    if not isinstance(im, np.ndarray):
        raise RuntimeError("Need to provide a numpy array, image has type " + str(type(im)))
    if not len(im.shape) == 2:
        raise RuntimeError("Need to provide an array with shape (n, m). Provided array has shape " + str(im.shape))
    if not _check_numpy_array_type(im):
        raise RuntimeError("Provided image has unsuported type: " + str(im.dtype))
    if np.any(np.isnan(im)):
        raise RuntimeError("Data contains a nan, decide how to handle missing data")
    if np.any(np.isinf(im)):
        raise RuntimeError("Data contains an np.inf, decide how to handle infinite values")
    if np.isclose(im.max(), 0) and np.isclose(im.min(), 0):
        raise RuntimeError("Inputed image is near to zero for all values")
    if np.isclose((im.max() - im.min()), 0):
        raise RuntimeError("Inputed image has nearly the same value for all pixels. Check input")

def _check_numpy_array_type(im):
    im = np.array(im)
    check_type = str(im.dtype)
    
    ok_types = ('int8',
                'int16',
                'int32',
                'int64',
                'uint8',
                'uint16',
                'uint32',
                'uint64',
                'float16',
                'float32',
                'float64')
    
    if check_type in ok_types:
        return True
    
    return False
    
def _check_numeric_types(number):
    if (isinstance(number, float)       or
        isinstance(number, int)         or
        isinstance(number, np.int)      or
        isinstance(number, np.int32)    or
        isinstance(number, np.int64)    or
        isinstance(number, np.float)    or
        isinstance(number, np.float16)  or
        isinstance(number, np.float32)  or
        isinstance(number, np.float64)  or
        isinstance(number, np.float128) ):
        return True
    return False

def _check_int_types(number):
    if (isinstance(number, int)         or
        isinstance(number, np.int)      or
        isinstance(number, np.int32)    or
        isinstance(number, np.int64)    or
        isinstance(number, np.uint8)    or
        isinstance(number, np.uint16)   or
        isinstance(number, np.uint32)   or
        isinstance(number, np.uint64)):
        return True
    return False

def _check_numpy_array_int_types(im):
    im = np.array(im)
    check_type = str(im.dtype)
    
    ok_types = ('int8',
                'int16',
                'int32',
                'int64',
                'uint8',
                'uint16',
                'uint32',
                'uint64')
    
    if check_type in ok_types:
        return True
    
    return False

def _check_numpy_array_string_types(ar):
    check_type = str(ar.dtype)
    
    ok_types = ('<U9',
                '<U7')
    
    if check_type in ok_types:
        return True
    
    return False

def _check_array_like(ar):

    if (isinstance(ar, type((1, 2))) or
        isinstance(ar, type([1, 2])) or
        isinstance(ar, type(np.array([1,2])))
       ):
        return True
    return False

def _check_function_input(im, thresh_func, args):
    
    if not callable(thresh_func):
        raise RuntimeError("The provided function is not callable")
        
    func_out = thresh_func(im, *args)
    if not (_check_numeric_types(func_out) or isinstance(func_out, np.ndarray)):
        raise RuntimeError("The provided function must output a numeric or array \
                           provided function returns type " + str(type(func_out)))
    if isinstance(func_out, np.ndarray) and not func_out.shape == im.shape:
        raise RuntimeError("Array output of the function must have same shape as the image \
                           the output array has shape " + str(func_out.shape) +
                           ", image has shape " + str(im.shape))

    return True

def _check_ims_same_dim(im1, im2):
    if not (im1.shape == im2.shape):
        raise RuntimeError("The provided images have different dimension \
        im1: "+str(im1.shape)+", im2: "+str(im2.shape))
        
def _check_roi_inputs(roi_kind, cent, width, height, outside_roi):
    
    if not _check_array_like(roi_kind):
        raise RuntimeError("The given roi kind object is not array like, it is " + str(type(roi_kind)))
    elif not _check_array_like(cent):
        raise RuntimeError("The given roi centers object object is not array like, it is " + str(type(cent)))
    elif not _check_array_like(width):
        raise RuntimeError("The given width object object is not array like, it is " + str(type(width)))
    elif not _check_array_like(height):
        raise RuntimeError("The given height object object is not array like, it is " + str(type(height)))
    
    if not _check_numpy_array_int_types(cent):
        raise RuntimeError("The cent object must have entries of integer type")
    elif not _check_numpy_array_int_types(width):
        raise RuntimeError("The width object must have entries of integer type")
    elif not _check_numpy_array_int_types(height):
        raise RuntimeError("The height object must have entries of integer type")
    elif not _check_numpy_array_string_types(np.array(roi_kind)):
        raise RuntimeError("The roi_kind object must have entries of type str")
          
def _check_crop_inputs(cent, width, height):    
    if not _check_array_like(cent):
        raise RuntimeError("The given cent object is not array like, it is " + str(type(cent)))
    if not _check_numpy_array_int_types(cent):
        raise RuntimeError("The cent object must have entries of integer type")
    if not _check_int_types(width):
        raise RuntimeError("The width must be integer type")
    if not _check_int_types(height):
        raise RuntimeError("The height must be integer type")

def region_props_to_tuple(rp):
    """Function to extract the region properties from a regionprops
    object.
    
    Parameters
    ----------
    rp : skimage.measure._regionprops._RegionProperties, a region properties
        object from which to extract attributes
    Returns
    -------
    attributes : tuple. The regionproperty fields for many properites of a blob
    labels : tuple. The corresponding label for teh values in attributes
    """
    
    if not isinstance(rp, skimage.measure._regionprops._RegionProperties):
        raise RuntimeError("Must provide a regionprops object returned by skimage.measure.regionprops")
        
    labels = ('area', 'bbox_min_row', 'bbox_min_col', 'bbox_max_row', 'bbox_max_col', 'bbox_area',
              'centroid_row', 'centroid_col', 'convex_area', 'eccentricity', 'equivalent_diameter',
              'euler_number', 'extent', 'filled_area', 'label', 'local_centroid_row', 'local_centoid_col',
              'major_axis_length', 'max_intensity', 'mean_intensity', 'min_intensity', 'minor_axis_length',
              'orientation', 'perimeter', 'solidity', 'weighted_centroid_row', 'weighted_centroid_col',
              'weighted_local_centoid_row', 'weighted_local_centroid_col')
    
    bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = rp.bbox
    centroid_row, centroid_col = rp.centroid
    local_centroid_row, local_centoid_col = rp.local_centroid
    weighted_centroid_row, weighted_centroid_col = rp.weighted_centroid
    weighted_local_centoid_row, weighted_local_centroid_col = rp.weighted_local_centroid

    attributes = (rp.area, bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col,
                  rp.bbox_area, centroid_row, centroid_col, rp.convex_area, 
                  rp.eccentricity, rp.equivalent_diameter, rp.euler_number,
                  rp.extent, rp.filled_area, rp.label, local_centroid_row,
                  local_centoid_col, rp.major_axis_length, rp.max_intensity,
                  rp.mean_intensity, rp.min_intensity, rp.minor_axis_length,
                  rp.orientation, rp.perimeter, rp.solidity,
                  weighted_centroid_row, weighted_centroid_col,
                  weighted_local_centoid_row, weighted_local_centroid_col)
    
    return attributes, labels

def construct_bg_img(ims, num_ims=10, quart=75):
    """Function to make a background image for a behavioral arena
    based on images of the arena populated by animals of interest.
    
    Parameters
    ----------
    ims : array like of np.array with shape (l, n, m)
        A set of image (with only one color chanel) to construct
        a background image with
    num_ims : int
        number of images to use in the construction of the background.
        If the animals in the arena do not move much, this number will need
        to be larger than the default of 10.
    quart : numeric
        the quartile of pixel values to decide what the background versus animal
        is. In an arena which is bottom lit, a quartile above 50 should give the
        value from the background (since the animal is dark and the bg is light)
    Returns
    -------
    output : 2d numpy.ndarray with shape (n, m)
        background image constructed from the data
    """
    
    if not _check_array_like(ims):
        raise RuntimeError('Provided ims object is not array like, it is type ' + str(type(ims)))
    
    [_check_image_input(im) for im in ims]
    
    if not _check_int_types(num_ims):
        raise RuntimeError('Please provide an integer for the num_ims parameter. Provided argument has type ' + str(type(num_ims)))
    
    rands = np.random.randint(low=0, high=len(ims), size=num_ims)
    w = ims[0].shape[0]
    h = ims[0].shape[1]
    l = len(rands) 
    ims_draw = np.zeros(shape=(l, w, h))
    for i, ind in enumerate(rands):
        ims_draw[i] = ims[ind]
                    
    new_im = np.zeros(ims[0].shape)
    for i in range(new_im.shape[0]):
        for j in range(new_im.shape[1]):
            new_im[i, j] = int(np.percentile(ims_draw[:, i, j], quart)) 
            
    return new_im
    
def bg_subtract(im1, im2):
    """Function to perform background subtraction on an image
    using a blank image of the arena of interest.
    
    Parameters
    ----------
    im1 : im, numpy.ndarray with shape (n, m) (with 0 < m, n)
        The image (with only one color chanel) to subtract the
        background from.
    im2 : im, numpy.ndarray with shape (n, m) (with 0 < m, n)
        The background image (with only one color chanel) to subtract the
        from im1.
    Returns
    -------
    output : 2d numpy.ndarray with shape (n, m)
        image with background subtracted, i.e. im1-im2.
    """
    
    _check_image_input(im1)
    _check_image_input(im2)
    _check_ims_same_dim(im1, im2)
    
    im1 = normalize_convert_im(im1)
    im2 = normalize_convert_im(im2)
    
    im_no_bg = im1-im2
    im_no_bg = normalize_convert_im(im_no_bg)
    return (im_no_bg)

def normalize_convert_im(im):
    """Function to normalize an image and convert it to float type.
    Normalized image is between 0. and 1.0.
    
    Parameters
    ----------
    im : numpy.ndarray with shape (n, m) (with 0 < m, n)
        The image (with only one color chanel) to get ROI from.
    
    Returns
    -------
    output : 2d numpy.ndarray with shape (n, m) of type float 
    and in range of 0. to 1.0.
    """
    
    _check_image_input(im)
    
    im = skimage.img_as_float(im)
    im_norm = im - im.min()
    im_norm = im_norm / im_norm.max()
    return im_norm

def crop_image_rectangle(im, cent=(50, 50), width=50, height=50):
    """Function to return a rectangularly cropped image.
    
    Parameters
    ----------
    im : numpy.ndarray with shape (n, m) (with 0 < m, n)
        The image (with only one color chanel) to get ROI from.
    cent : array like integer pairs for the center of subimage region to keep
    width : widths of cropping region from the center point given above to the edge.
    height : height of cropping region from the center point to the top or bottom.
    
    Returns
    -------
    output : 2d numpy.ndarray with shape (2*width, 2*height)
    """
    
    
    _check_image_input(im)
    _check_crop_inputs(cent, width, height)
    
    nrows, ncols = im.shape
    centx, centy = cent
    centy = nrows - centy
    
    y1, y2 = (max(centy-height, 0), centy+height)
    x1, x2 = (centx-width, centx+width)
    im_cropped = im[y1:y2, x1:x2]
    
    return im_cropped

def give_roi_of_image(im, roi_kind=('rectangle',), cent=((50, 50)), width=(50,), height=(50,), outside_roi='max'):
    """Function to return an image with area outside of an ROI set to a value.
    
    Parameters
    ----------
    im : numpy.ndarray with shape (n, m) (with 0 < m, n)
        The image (with only one color chanel) to get ROI from.
    roi_kind : array like of strings for ROI types
    cent : array like of array like integer pairs for the center of ROIs
    width : array like with widths of ROIs (from rectangle/circle center to left or right edge).
        Represents horizontal axis (major or minor) in the case of 'circle' mode of ROI.
    height : array like with heights of ROIs (from rectangle/circle center to top or bottom).
        Represents vertical axis (major or minor) in the case of 'circle' mode of ROI.
    outside_roi : either 'max', 'min' or a value. This gives what to set
        the image region outside of the ROI.
    
    Returns
    -------
    output : 2d numpy.ndarray with shape (n, m)
        array with image parts in ROI unaltered and 
        outside ROI set to 'outside_roi'
    """
    
    _check_image_input(im)
    _check_roi_inputs(roi_kind, cent, width, height, outside_roi)
    
    nrows, ncols = im.shape
    row, col = np.ogrid[:nrows, :ncols]
    outer_disk_mask = row + col > -1
    
    for roi_k, cents, width, height in zip(roi_kind, cent, width, height):
        centx, centy = cents
        centy = nrows - centy
    
        if roi_k == 'ellipse':
            outer_disk_mask *= (((row - centy)/height)**2 + ((col - centx)/width)**2 > 1)
            
        elif roi_k == 'rectangle':
            y1, y2 = (max(centy-height, 0), centy+height)
            x1, x2 = (centx-width, centx+width)
            outer_disk_mask[y1:y2, x1:x2] = False
    
    im_roi = im.copy()
    if outside_roi == 'max':
        im_roi[outer_disk_mask] = im.max()
    
    return im_roi

def segment(im, thresh_func=skimage.filters.threshold_otsu, args=()):
    """Function to threshold an image using skimage functions. 
    The user pases the desired function to determine the threshold
    for the data (or a value to use as the threshold). This value
    for the threshold level can be a float/int or an array of same shape
    as the input image.
    
    Parameters
    ----------
    x : im, numpy.ndarray with shape (n, m) (with 0 < m, n)
        The image (with only one color chanel) to threshold.
    thresh_func : function
        The function to use to calculate the thresholding. Should
        return a single scalar value or a numpy array.
    Returns
    -------
    im_bw : 2d numpy.ndarray with shape (n, m)
        Boolean array with location of thresholded objects.
    im_labeled : a labelfield of image
    n_labels : number of identified objects in the labelfield
    """
    
    _check_image_input(im)
    if not (_check_numeric_types(thresh_func)):
        _check_function_input(im, thresh_func, args)
        thresh = thresh_func(im, *args)
    else:
        thresh = thresh_func
    
    im_bw = im < thresh
    
    # Label binary image; background kwarg says value in im_bw to be background
    im_labeled, n_labels = skimage.measure.label(im_bw, background=0, return_num=True)
    
    return im_bw, im_labeled, n_labels
    