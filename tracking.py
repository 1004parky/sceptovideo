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

def process_im(im, im_bg=None, size_thresh=[0, 0], props='soc'):
    '''Function to identify properties of blobs of interest (BOI) from an image
    
    PARAMETERS
    ----------
    im: numpy.ndarray, a 2d array representing an image with BOI
    size_thresh: list, a list with the lower and upper size bounds for a BOI.  
    Default treats all blobs as BOI.  
    im_bg: numpy.ndarray, a 2d array representing the image with no BOI
    props: string of keyword characters for properties to extract about BOI.  
    Default values are s(size), o(orientation), and c(centroid).  
    Valid options:
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

    # Error handling
    if len(size_thresh) != 2:
        raise Exception('size_thresh must have a lower and upper size limit')
    if type(im) != np.ndarray:
        raise Exception('im must be a 2d numpy ndarray.')
    if len(props) != len(set(props)):
        print('Warning: Some property keywords are repeated. Redundancies were removed.')
        props = list(set(props))
    for x in props:
        if x not in _kw_dict:
            raise Exception('Property keyword "'+x+'" not recognized. See documentation.')

    if size_thresh == [0, 0]:
        thresholding = False
    else:
        thresholding = True

    if im_bg:
        if type(im_bg) != np.ndarray:
            raise Exception('im_bg must be a 2d numpy ndarray.')
        im = seg.bg_subtract(im, im_bg)
    im_bw, im_labeled, n_labels = seg.segment(im)

    #spec_labels is a list of all specimen [label, size, coordinates]
    spec_labels = []
    properties = skimage.measure.regionprops(im_labeled)

    for i in range(0, n_labels):
        if thresholding:
            if properties[i]['area'] not in range(size_thresh[0], size_thresh[1]):
                continue
        spec_labels.append([ properties[i][_kw_dict[x]]  for x in props])
            
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
        metrics = list(set(metrics))

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


def track(fpath, bg_path=None, size_thresh=[0, 0], metrics='soc', 
          weights=None, same_weights=True, 
          dist_func=scipy.spatial.distance.euclidean):
    '''Tracks the location of several blobs of interest across multiple images

    PARAMETERS
    ----------
    fpath: str, absolute path of the directory containing images
    bg_path: str, absolute path of the file
    size_thresh: list, a list with the lower and upper size bounds for a BOI.  
    Default treats all blobs as BOI. 
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

    data = pd.DataFrame()
    im_data = []

    for x in all_files:
        im_test = skimage.io.imread(x)
        im_data.append(process_im(im_test, im_bg=im_bg, size_thresh=size_thresh))
    
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