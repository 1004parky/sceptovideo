import numpy as np
import pandas as pd
import glob

import skimage.io
import skimage.morphology
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
    blob: np.ndarray with just one blob
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
        print('WARNING: Some metrics keywords are repeated. Redundancies were removed.')
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


def process_im(im, im_bg=None, props='soc', include_frame=False, frame_id = 0, 
               sep_centroid = False, min_size = 200, filter_func=thresholding, *args, **kwargs):
    '''Function to identify properties of blobs of interest (BOI) from an image
    
    PARAMETERS
    ----------
    im: numpy.ndarray, a 2d array representing an image with BOI
    thresh_list: numpy.ndarray with lower and upper limits (col) for each 
    metric (row).  Default treats all blobs as BOI.  
    im_bg: numpy.ndarray, a 2d array representing the image with no BOI
    props: string of keyword characters for properties to extract about BOI.  
    Default values are s(size), o(orientation), and c(centroid).  
    include_frame: bool, True includes frame number in returned DataFrame
    frame_id: frame number. Only relevant if include_frame is True
    sep_centroid: True separates centroid into x and y columns
    min_size: int, min size to be used by skimage.morphology.remove_small_objects func
    filter_func: function name used to filter blobs into BOI.  
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
        print('WARNING: Some property keywords are repeated. Redundancies were removed.')
        props = list(sorted(set(props), key=props.index))
    if not callable(filter_func) and (filter_func != None):
        raise Exception('Filter function is not callable.')
    if type(include_frame) != bool:
        raise Exception('include_frame must be a boolean.')
    if type(im_bg) != np.ndarray:
        if im_bg == None:
            pass
        else:
            raise Exception('im_bg must be a 2d numpy ndarray.')
    if sep_centroid and 'c' not in props:
        raise Exception('There is no centroid data to separate.')

    if type(im_bg) == np.ndarray:
        im = seg.bg_subtract(im, im_bg)
    im_bw, im_labeled, n_labels = seg.segment(im, thresh_func = skimage.filters.threshold_local, args =[25])
    im = skimage.morphology.remove_small_objects(im_labeled, min_size=min_size, connectivity=1) + 0
    
    im_labeled, n_labels = skimage.measure.label(im, return_num=True)

    #specs is a list of all specimen [label, size, coordinates]
    specs = []
    properties = skimage.measure.regionprops(im_labeled)

    for i in range(n_labels):
        if filtering:
            blob = (im_labeled == (i+1)) + 0
            if filter_func(blob, *args, **kwargs):
                spec_data = [properties[i][_kw_dict[x]] for x in props]
                if include_frame:
                    spec_data.insert(0, frame_id)
            else:
                continue
        else:
            spec_data = [properties[i][_kw_dict[x]] for x in props]
            if include_frame:
                spec_data.insert(0, frame_id)
        specs.append(spec_data)

    col_names = [_kw_dict[x] for x in props]
    if include_frame:
        col_names.insert(0, 'frame')

    df = pd.DataFrame(data=specs, columns=col_names)
    if sep_centroid:
        df = df.reindex(columns=col_names + ['x', 'y'])
        df[['x', 'y']] = pd.DataFrame(df['centroid'].values.tolist()) #df['centroid'].apply(pd.Series)
        df.drop('centroid', axis = 1, inplace = True)

    return df


def match_blobs(oim, nim, data=pd.DataFrame(), metrics='soc', weights=None, 
                same_weights=True, sep_centroid=False, sequential=True, 
                dist_func=scipy.spatial.distance.euclidean):
    '''Function to match blobs across two images using user-specified metrics

    PARAMETERS
    ----------
    oim: pandas.DataFrame with processed image data (see function process_im)
    nim: pandas.DataFrame representing one timepoint after oim
    data: pandas.DataFrame, stores centroids for each blob (col) and timepoint (row).  
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
    sep_centroid: True separates centroid into x and y columns
    sequential: Assumes that nim is the frame after oim, assigns frame numbers accordingly.  
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

    if data.empty:
        to_add = []
        for i in range(oim.shape[0]):
            to_add.append([oim[_kw_dict[m]].values[i] for m in metrics])
            try:
                to_add[i].append(oim['frame'].values[i])
            except:
                oim['frame'] = [0 for o in range(oim.shape[0])]
                to_add[i].append(oim['frame'].values[i])
            to_add[i].append(i)
            oim.at[i, 'id'] = i
        col_names = [_kw_dict[m] for m in metrics]
        col_names.append('frame')
        col_names.append('id')
        data = pd.DataFrame(data=to_add, columns=col_names)

    # Error handling
    if sep_centroid and 'c' not in metrics:
        raise Exception('There is no centroid data to separate.')
    try:
        oim['frame']
        try:
            nim['frame']
            if (not sequential) and (oim['frame'][0] == nim['frame'][0]):
                print('WARNING: Frame numbers for oim and nim are the same.')
        except: 
            pass
    except:
        pass
    if not type(data) == pd.core.frame.DataFrame:
        raise Exception('data must be a pandas.DataFrame.')
    try:
        data['frame']
    except:
        raise Exception('data must contain frame column')
    try:
        data['id']
    except:
        raise Exception('data must contain id column')
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
        print('WARNING: Some property keywords were repeated. Redundancies were removed')
        metrics = list(sorted(set(metrics), key=metrics.index))
    if nim.shape[0] == 0:
        return data
    # Initialize matrix of comparison scores
    comp_matrix = np.zeros((nim.shape[0], oim.shape[0], len(metrics)))

    # Fill in comparison matrix
    for i in range(len(metrics)):
        for n in range(nim.shape[0]):
            for o in range(oim.shape[0]):
                if metrics[i] == 'c':
                    comp_matrix[n][o][i] = (weights[i] * 
                        dist_func(oim[_kw_dict[metrics[i]]].values[o], 
                        nim[_kw_dict[metrics[i]]].values[n]))
                elif metrics[i] != 'y':
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
                        for j in range(oim.shape[0])] 
                        for i in range(nim.shape[0])])
    
    new_labels = np.argmin(p_match, axis=1)

    max_id = max(data['id'].values)
    new_id = np.array([int(oim['id'].values[i]) for i in new_labels])

    # If more than one nim blob matched to oim blob, identify true match and
    # assign new id to other nim blobs.
    for i in range(len(new_labels)):
        if (new_labels == new_labels[i]).sum() > 1:
            rpt = [j for j, x in enumerate(new_labels) if x == new_labels[i]]
            match_poss = [p_match[k][new_labels[i]] for k in rpt]
            true_match = rpt[np.argmin(match_poss)]
            for l in rpt:
                if l != true_match:
                    new_id[l] = max_id
                    max_id += 1

    to_add = []
    for i in range(nim.shape[0]):
        to_add.append([nim[_kw_dict[m]].values[i] for m in metrics])
        if sequential:
            nim['frame'] = [oim['frame'].values[0]+1 for n in range(nim.shape[0])]
            to_add[i].append(nim['frame'].values[i])
        else:
            try:
                to_add[i].append(nim['frame'].values[i])
            except:
                nim['frame'] = [oim['frame'].values[0]+1 for n in range(nim.shape[0])]
                to_add[i].append(nim['frame'].values[i])
        to_add[i].append(new_id[i])
    col_names = [_kw_dict[m] for m in metrics]
    col_names.append('frame')
    col_names.append('id')
    new_tp = pd.DataFrame(data=to_add, columns=col_names)

    data = pd.concat([data, new_tp], axis=0, ignore_index=True)
    if sep_centroid:
        data[['x', 'y']] = data['centroid'].apply(pd.Series)
        data.drop('centroid', axis = 1, inplace = True)
    
    return data, new_tp


def track(all_files, bg_path=None, metrics='soc', weights=None, same_weights=True, 
          dist_func=scipy.spatial.distance.euclidean, filter_func=thresholding, *args, **kwargs):
    '''Tracks the location of several blobs of interest across multiple images

    PARAMETERS
    ----------
    all_files: list, absolute paths of images to be processed
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
    metrics = list(metrics)
    if same_weights and (weights == None):
        weights = [1] * len(metrics)

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

    for x in range(len(all_files)):
        im_test = skimage.io.imread(all_files[x])
        im_data.append(process_im(im_test, im_bg=im_bg, include_frame = True, 
                       frame_id=x, filter_func=filter_func, *args, **kwargs))

    to_add = []
    for i in range(im_data[0].shape[0]):
        to_add.append([im_data[0][_kw_dict[m]].values[i] for m in metrics])
        try:
            to_add[i].append(im_data[0]['frame'].values[i])
        except:
            im_data[0]['frame'] = [0 for o in range(im_data[0].shape[0])]
            to_add[i].append(im_data[0]['frame'].values[i])
        to_add[i].append(i)
        im_data[0].at[i, 'id'] = i
    col_names = [_kw_dict[m] for m in metrics]
    col_names.append('frame')
    col_names.append('id')
    data = pd.DataFrame(data=to_add, columns=col_names)
    
    for j in range(len(im_data)-1):
        data, im_data[j+1] = match_blobs(im_data[j], 
                                         im_data[j+1], 
                                         data,
                                         metrics=metrics, 
                                         weights=weights,
                                         same_weights=same_weights, 
                                         dist_func=dist_func)
    
    if 'c' in metrics:
        data[['x', 'y']] = data['centroid'].apply(pd.Series)
        data.drop('centroid', axis = 1, inplace = True)
    return data