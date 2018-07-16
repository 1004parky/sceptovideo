import numpy as np
import pandas as pd

#Image processing
import skimage.filters
import skimage.io
import skimage.morphology
import skimage.exposure
import skimage.feature

import bootcamp_utils

from behavioral_analysis import segmentation as seg

#Takes an image and the bg image and returns the coordinates and sizes of the blobs of interest.
def processImage(im, im_bg, thresh):
    im_no_bg=seg.bg_subtract(im,im_bg)
    im_bw, im_labeled, n_labels = seg.segment(im_no_bg)
    #Filter blobs by size. Not too large, not too small.

    #spec_labels is a list of all specimen [label, size, coordinates]
    spec_labels=[]
    properties=skimage.measure.regionprops(im_labeled)
    
    for i in range(0,n_labels):
        if properties[i].area>thresh[0] and properties[i].area<thresh[1]: #about the size of ants/beetles
            spec_labels.append([properties[i].area, properties[i].orientation, properties[i].centroid])
            
    return pd.DataFrame(data=spec_labels, columns=['size','orientation','centroid'])

def matchBlobs(oim, nim, db, size_weight=1, orient_weight=1, xy_weight=1):
    #oim is old image, nim is new image. Match nim indices to oim indices.
    #db is a dataframe of timepoints (rows) by specimen (column) filled with centroids.
    assert oim.shape==nim.shape
    assert type(db)==pd.core.frame.DataFrame

    #number of specimen
    n_spec=oim.shape[0]

    #Matrix of comparison scores 
    sizediff=[[np.nan for i in range(n_spec)] for j in range(n_spec)]
    orientdiff=[[np.nan for i in range(n_spec)] for j in range(n_spec)]
    xydiff=[[np.nan for i in range(n_spec)] for j in range(n_spec)]

    #calculate diff in size, orientation, and centroids for every combination of new blob and old blob
    for n in range(n_spec):
        for o in range(n_spec):
            sizediff[n][o]=size_weight*(abs(oim.get('size')[o]-nim.get('size')[n]))/oim.get('size')[o]
            orientdiff[n][o]=orient_weight*abs(oim.get('orientation')[o]-nim.get('orientation')[n])
            dist_vec=np.subtract(oim.get('centroid')[o],nim.get('centroid')[n])
            xydiff[n][o]=xy_weight*np.sqrt(np.dot(dist_vec,dist_vec)) #dot product of difference vector

    #Add the differences up, want to minimize the sum of differences.
    matchProb=np.add(np.add(sizediff,orientdiff),xydiff)

    nlabels=np.argsort(np.argmin(matchProb, axis=1))    
    
    #List of centroids in new order.
    ncentroids=[nim.get('centroid')[k] for k in nlabels]

    new_tp=pd.Series(data=ncentroids)
    return pd.concat([db,new_tp], axis=1, ignore_index=True), nim.reindex(nlabels)

"""
def getDisp():
    #Get displacement, just take first and last point.

def getDist():
    #Add up all distances traveled.
    
def getSpeed():
    #Return a list of speeds
    
def getSociality():
    #if any specimen within a certain radius of any other specimen, count as social behavior?
    
def trackImgs(fpath, bg_path, thresh)
    all_files=glob.glob(fpath)
    im_bg=skimage.io.imread(bg_path)

    data=pd.DataFrame()
    processed_im=[]

    for x in test_files:
        im_test=skimage.io.imread(x)
        processed_im.append(processImage(im_test,im_bg,thresh))

    data=pd.concat([data, processed_im[0].get('centroid')], axis=1, ignore_index=True)
    for j in range(len(processed_im)-1):
        data, processed_im[j+1]=matchBlobs(processed_im[j], processed_im[j+1], data)
"""