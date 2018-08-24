import numpy as np
import pandas as pd
import skimage.draw
import random

def rand_data(arena, n_shapes, maj_axis, min_axis):
    '''Generates an image with randomly scattered, not overlapping ellipses 
    
    PARAMETERS
    ----------
    arena: ndarray of 1's and 0's, with 0's where shapes should be generated.
    n_shapes: int, number of shapes to be generated.
    maj_axis: int, length of major axis of ellipse
    min_axis: int, length of minor axis of ellipse
    
    RETURNS
    -------
    im: ndarray of 1's and 0's with 1's where shapes are.
    centroids: dataframe of centroids
    '''
    # Error handling
    if type(arena) != np.ndarray:
        raise Exception('arena must be a numpy ndarray.')
    for r in arena:
        for c in r:
            if c != 0 and c != 1: 
                raise Exception('arena must contain only ones and zeros.')
    if type(n_shapes) != int:
        raise Exception('n_shapes must be an integer.')
    if type(maj_axis) != int:
        raise Exception('maj_axis must be an integer.')
    if type(min_axis) != int:
        raise Exception('min_axis must be an integer.')
    if n_shapes <= 0:
        raise Exception('n_shapes must be a postive integer.')
    
    im = arena
    n_drawn = 0
    attempt = 0
    centroids = pd.DataFrame(columns = list('xy'))
    
    while n_drawn < n_shapes:
        # Start coordinates
        row = random.randint(0, im.shape[0]-1)
        col = random.randint(0, im.shape[1]-1)
        # Check that centroid is within arena.
        if im[row, col] == 1:
            continue
        point = pd.DataFrame(data = [(row, col)], columns = list('xy'))
        centroids = pd.concat([point, centroids], ignore_index=True)
        orient = random.uniform(-1 * np.pi, np.pi)
        try:
            rr, cc = skimage.draw.ellipse(row, col, min_axis, maj_axis, rotation=orient)
            blob = np.ndarray((im.shape[0], im.shape[1]))
            blob[rr, cc] = 1
        except IndexError:
            continue
        tmp = np.add(im, blob)
        if True in (tmp > 1):
            #Blob had overlap
            if attempt >= 1000:
                print(tmp[rr, cc])
            attempt += 1
            continue
        else:
            im = tmp
            n_drawn += 1
            attempt = 0
    return im, centroids


def clustering(data, mult_frame = True, exclude_extreme=False):
    '''Measures clustering given a dataframe of centroids.
    
    PARAMETERS
    ----------
    data: pd.DataFrame, contains all the centroids of the blobs of interest
    mult_frame: bool, default True. False means that data only contains info for one frame.
    exclude_extreme: bool, default False. True removes two most extreme values 
    from sociality calculation
    
    RETURNS
    -------
    spread: list, sum of the distances between the individual centroids and the center point
    for each frame
    '''
    if not mult_frame:
        data['frame'] = [0 for i in range(data.shape[0])]
        
    #Error handling
    if type(data) != pd.core.frame.DataFrame:
        raise Exception('data must be a pandas DataFrame.')
    for c in ['x', 'y', 'frame']:
        if c not in data.columns:
            raise Exception('data must contain x, y, and frame as columns.')
    
    frames = data.frame.unique()
    spread = [0 for i in range(len(frames))]
    for f in range(len(frames)):
        one_frame = data[data.frame == frames[f]]
        centroids = [[one_frame['x'].values[i], one_frame['y'].values[i]]
                     for i in range(one_frame.shape[0])]
        center = np.mean(centroids, axis=0)
        diff = [np.subtract(c, center) for c in centroids]
        dist = [np.sqrt(np.dot(d, d)) for d in diff]
        if exclude_extreme:
            ind_max = np.argmax(dist)
            ind_min = np.argmin(dist)
            del centroids[ind_max]
            del centroids[ind_min]
            #Recalculate
            center = np.mean(centroids, axis=0)
            diff = [np.subtract(c, center) for c in centroids]
            dist = [np.sqrt(np.dot(d, d)) for d in diff]
        spread[f] = np.mean(dist)
    return spread


def sociality(data, mult_frame=True, radius=0.0):
    '''Measures sociality of each blob in a frame for several frames.  Sociality of one blob 
    is defined as being within a certain radius from at least one other blob. 
    
    PARAMETERS
    ----------
    data: pd.DataFrame, contains all the centroids of the blobs of interest, sorted by frame
    mult_frame: bool, default True. False means that data only contains info for one frame.
    radius: float, limit for how close blobs must be to count as "social."  Default of 0 will
    calculate radius automatically from average blob size.

    RETURNS
    -------
    sociality: list, measure of sociality 
    '''
    if not mult_frame:
        data['frame'] = [0 for i in range(data.shape[0])]
    if radius == 0.0:
        try:
            all_rad = []
            for i in range(data.shape[0]):
                all_rad += np.mean([data['major_axis_length'].values[i], data['minor_axis_length'].values[i]])
            radius = np.mean(all_rad)
        except:
            raise Exception('major_axis_length and minor_axis_length not defined.')
    
    #Error handling
    if type(data) != pd.core.frame.DataFrame:
        raise Exception('data must be a pandas DataFrame.')
    if type(radius) != float:
        try:
            radius = float(radius)
        except:
            raise Exception('radius must be a float.')
    for c in ['x', 'y', 'frame']:
        if c not in data.columns:
            raise Exception('data must contain x, y, and frame as columns.')
    
    frames = data.frame.unique()
    sociality = [0 for i in range(len(frames))]
    for f in range(len(frames)):
        one_frame = data[data.frame == frames[f]]
        n_blob = one_frame.shape[0]
        dist = np.ndarray((n_blob, n_blob))
        for i in range(n_blob-1):
            c1 = [one_frame['x'].values[i], one_frame['y'].values[i]]
            c2 = [one_frame['x'].values[i+1], one_frame['y'].values[i+1]]
            d = np.subtract(c1, c2)
            dist[i][i+1] = np.sqrt(np.dot(d, d))
            dist[i+1][i] = dist[i][i+1]
        #Get rid of diagonal
        social = (dist < radius)
        social = np.triu(social, k=1) + np.tril(social, k=-1)
        sociality[f] = np.mean(np.sum(social, axis=0))
    return sociality