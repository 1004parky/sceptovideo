import numpy as np
import pandas as pd
import glob

import skimage.io
import scipy.spatial.distance

import bootcamp_utils
from behavioral_analysis import segmentation as seg

_kw_dict = {'c': 'centroid',
            'e': 'eccentricity',
            'j': 'major_axis_length',
            'n': 'minor_axis_length',
            'o': 'orientation',
            'p': 'perimeter',
            's': 'area'}


def thresholding(blob, thresh_list=[(0, 0), (0, 0)], metrics='se'):
    '''Determines whether a blob is a blob of interest (BOI).  

    PARAMETERS
    ----------
    blob: np.ndarray
    thresh_list: list of tuples with lower and upper bounds for each metric
    metrics: string of keyword characters for metrics to filter blobs into BOI.  
    Default values are s(size) and e(eccentricity)
    Valid options for metrics:
        | KEYWORD | MEANING           |
        | e       | eccentricity      |
        | j       | major_axis_length |
        | n       | minor_axis_length |
        | o       | orientation       |
        | p       | perimeter         |
        | s       | size              |
    
    RETURNS
    -------
    True if blob is a BOI, False if not.  
    '''
    metrics = list(metrics)

    if len(metrics) != len(set(metrics)):
        print('Warning: Some metrics keywords are repeated. Redundancies were removed.')
        metrics = list(sorted(set(metrics), key=metrics.index))    
    for m in metrics:
        if m not in _kw_dict:
            raise Exception('Metrics keyword "'+m+'" not recognized. See documentation.')
        if m == 'c':
            raise Exception('Thresholding functiion cannot process centroid data.')
    if len(metrics) != len(thresh_list):
        raise Exception('Every metric must have thresholds.')
    for i in range(len(thresh_list)):
        if type(thresh_list[i]) != tuple:
            raise Exception('Enter thresholds as a list of tuples.')
        if len(thresh_list[i]) != 2:    
            raise Exception('All thresholds must have a lower and upper size limit')
        if thresh_list[i] == (0, 0):
            raise Exception('Please enter thresholds.')
        if thresh_list[i][0] >= thresh_list[i][1]:
            raise Exception('Threshold error. Lower bound must be smaller than upper bound.')
    
    hit = np.zeros((len(metrics), 1))
    properties = skimage.measure.regionprops(blob)
    for i in range(len(metrics)):
        if (properties[0][_kw_dict[metrics[i]]] > thresh_list[i][0]) and (properties[0][_kw_dict[metrics[i]]] < thresh_list[i][1]):
            hit[i] = 1
    return (0 not in hit)


def process_im(im, im_bg=None, props='soc', filter_func=thresholding, *args, **kwargs):
    '''Function to identify properties of blobs of interest (BOI) from an image
    
    PARAMETERS
    ----------
    im: numpy.ndarray, a 2d array representing an image with BOI
    thresh_list: numpy.ndarray with lower and upper limits (col) for each 
    metric (row).  Default treats all blobs as BOI.  
    im_bg: numpy.ndarray, a 2d array representing the image with no BOI
    props: string of keyword characters for properties to extract about BOI.  
    Default values are s(size), o(orientation), and c(centroid).  
    filter_function: function name used to filter blobs into BOI.  
    First argument must take a boolean numpy.ndarray (one blob).  
    Default is tracking.thresholding function.  
    Use None to consider all blobs as BOI.  
    *args: args for the given filter_function.  
    **kwargs: kwargs for the given filter_function.
    Valid options for props:
        | KEYWORD | MEANING           |
        | c       | centroid          |
        | e       | eccentricity      |
        | j       | major_axis_length |
        | n       | minor_axis_length |
        | o       | orientation       |
        | p       | perimeter         |
        | s       | size              |
    
    RETURNS
    -------
    pandas.DataFrame of BOIs (rows) and desired attributes (cols)
    '''
    props = list(props)
    if filter_func != None:
        filtering = True

    # Error handling
    if type(im) != np.ndarray:
        raise Exception('im must be a 2d numpy ndarray.')
    if len(props) != len(set(props)):
        print('Warning: Some property keywords are repeated. Redundancies were removed.')
        props = list(sorted(set(props), key=props.index))
    if not callable(filter_func) and (filter_func != None):
        raise Exception('Filter function is not callable.')
    if im_bg:
        if type(im_bg) != np.ndarray:
            raise Exception('im_bg must be a 2d numpy ndarray.')
        im = seg.bg_subtract(im, im_bg)
    im_bw, im_labeled, n_labels = seg.segment(im)

    #spec_labels is a list of all specimen [label, size, coordinates]
    spec_labels = []
    properties = skimage.measure.regionprops(im_labeled)

    if filtering:
        for i in range(n_labels):
            blob = (im_labeled == (i+1)) + 0
            if filter_func(blob, *args, **kwargs):
                spec_labels.append([properties[i][_kw_dict[x]] for x in props])
    else:
        spec_labels.append([properties[i][_kw_dict[x]] for x in props] for i in range(n_labels))

    return pd.DataFrame(data=spec_labels, 
                        columns=[_kw_dict[x] for x in props])


def match_blobs(oim, nim, center_coord=pd.DataFrame(), metrics='soc', 
                weights=None, same_weights=True, 
                dist_func=scipy.spatial.distance.euclidean):
    '''Function to match blobs across two images using user-specified metrics

    PARAMETERS
    ----------
    oim: pandas.DataFrame with processed image data (see function process_im)
    nim: pandas.DataFrame representing one timepoint after oim
    center_coord: pandas.DataFrame, stores centroids for each blob (col) and timepoint (row).  
    Default creates a blank DataFrame.
    metrics: str of metric keywords with which to match blobs.  
    Defaults are s(size), o(orientation), and c(centroid distance). 
        Valid options:
        | KEYWORD | MEANING           |
        | c       | centroid distance |
        | e       | eccentricity      |
        | j       | major_axis_length |
        | n       | minor_axis_length |
        | o       | orientation       |
        | p       | perimeter         |
        | s       | size              |
    same_weights: boolean, True applies same weight (1) to all metrics
    weights: list of weights corresponding respectively to given metrics
    dist_func: scipy.spatial.distance function.  Default is euclidean.  
    Other options include .braycurtis, .canberra, .cityblock, etc.

    RETURNS
    -------
    pandas.DataFrame with centroids for newest timepoint (nim) added
    pandas.DataFrame, essentially nim with reordered rows to match oim's indexing
    '''
    metrics = list(metrics)
    if same_weights and (weights == None):
        weights = [1] * len(metrics)

    # Error handling
    if not oim.shape[0] == nim.shape[0]:
        raise Exception('oim and nim must have the same number of rows')
    if not type(center_coord) == pd.core.frame.DataFrame:
        raise Exception('center_coord must be a pandas.DataFrame.')
    if len(weights) != len(metrics):
        raise Exception('Must provide a weight for each metric provided')
    for x in metrics:
        if x not in _kw_dict:
            raise Exception('Metric keyword "'+x+'" not recognized. See documentation.')
        try:
            oim[(_kw_dict[x])]
        except:
            raise Exception('Old DataFrame does not contain data for metric '+_kw_dict[x])
        try:
            nim[(_kw_dict[x])]
        except:
            raise Exception('New DataFrame does not contain data for metric '+_kw_dict[x])
    if not callable(dist_func):
        raise Exception('Distance function is not callable.')
    if len(metrics) != len(set(metrics)):
        print('Warning: Some property keywords were repeated. Redundancies were removed')
        metrics = list(sorted(set(metrics), key=metrics.index))
    
    if center_coord.empty:
        for i in range(oim.shape[0]):  
            center_coord[i] = [oim['centroid'].values[i]]

    # Number of specimen
    n_spec = oim.shape[0]

    # Initialize matrix of comparison scores
    comp_matrix = np.zeros((n_spec, n_spec, len(metrics)))

    # Fill in comparison matrix
    for i in range(len(metrics)):
        for n in range(n_spec):
            for o in range(n_spec):
                if metrics[i] == 'c':
                    comp_matrix[n][o][i] = (weights[i] * 
                        dist_func(oim[_kw_dict[metrics[i]]].values[o], 
                                  nim[_kw_dict[metrics[i]]].values[n]))
                else:
                    try:
                        comp_matrix[n][o][i] = (weights[i] * 
                            abs(oim[_kw_dict[metrics[i]]].values[o] - 
                            nim[_kw_dict[metrics[i]]].values[n]) /
                            oim[_kw_dict[metrics[i]]].values[o])
                    except:
                        comp_matrix[n][o][i] = (weights[i] * 
                            abs(oim[_kw_dict[metrics[i]]].values[o] - 
                            nim[_kw_dict[metrics[i]]].values[n]))

    # Add the differences up, want to minimize the sum of differences.
    p_match = np.array([[sum(comp_matrix[i][j][:]) 
                        for j in range(n_spec)] for i in range(n_spec)])

    new_labels = np.argsort(np.argmin(p_match, axis=1))    
    
    # List of centroids in new order.
    n_centroids = [nim.get('centroid')[k] for k in new_labels]
    new_tp = pd.DataFrame()
    for i in range(n_spec):  
        new_tp[i] = [n_centroids[i][:]]

    return (pd.concat([center_coord, new_tp], axis=0, ignore_index=True), 
                      nim.reindex(new_labels))


def track(fpath, bg_path=None, metrics='soc', weights=None, same_weights=True, 
          dist_func=scipy.spatial.distance.euclidean, filter_func=thresholding, *args, **kwargs):
    '''Tracks the location of several blobs of interest across multiple images

    PARAMETERS
    ----------
    fpath: str, absolute path of the directory containing images
    bg_path: str, absolute path of the file
    metrics: str of metrics with which to match blobs.  
    Defaults are s(size), o(orientation), and c(centroid distance). 
        Valid options:
        | KEYWORD | MEANING           |
        | c       | centroid distance |
        | e       | eccentricity      |
        | j       | major_axis_length |
        | n       | minor_axis_length |
        | o       | orientation       |
        | p       | perimeter         |
        | s       | size              |
    same_weights: boolean, True applies same weight (1) to all metrics
    weights: list of weights corresponding respectively to given metrics
    dist_func: scipy.spatial.distance function.  Default is euclidean.  
    Other options include .braycurtis, .canberra, .cityblock, etc.
    filter_function: function name used to filter blobs into BOI.  
    First argument must take a boolean numpy.ndarray (one blob).  
    Default is tracking.thresholding function.  
    Use None to consider all blobs as BOI.  

    RETURNS
    -------
    data: pandas.DataFrame with centroids for every specimen (col) at each tp (row)
    '''
    if same_weights and (weights == None):
        weights = [1] * len(metrics)
    all_files = glob.glob(fpath)

    # Error handling
    if len(weights) != len(metrics):
        raise Exception('Must provide a weight for each metric provided')
    if len(all_files) == 0:
        raise Exception('Directory of images not found.')
    if bg_path:
        try:
            im_bg = skimage.io.imread(bg_path)
        except:
            raise Exception('Background image file not found')
    else:
        im_bg = None
    if not callable(dist_func):
        raise Exception('Distance function is not callable.')
    if not callable(filter_func):
        raise Exception('Filter function is not callable.')

    data = pd.DataFrame()
    im_data = []

    for x in all_files:
        im_test = skimage.io.imread(x)
        im_data.append(process_im(im_test, im_bg=im_bg, filter_func=filter_func, *args, **kwargs))
    
    first_tp = pd.DataFrame(data = [im_data[0].get('centroid').tolist()])
    data = pd.concat([data, first_tp], axis=0, ignore_index=True)

    for j in range(len(im_data)-1):
        data, im_data[j+1] = match_blobs(im_data[j], 
                                         im_data[j+1], 
                                         data,
                                         metrics=metrics, 
                                         weights=weights,
                                         same_weights=same_weights, 
                                         dist_func=dist_func)
    return data