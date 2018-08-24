import numpy as np
import pandas as pd
import os

#Image processing
import skimage.filters
import skimage.io
import skimage.morphology

import altair as alt
import bokeh.io

import bootcamp_utils

#Local functions created for blob tracking
from behavioral_analysis import segmentation as seg
import tracking
import sociality
import matplotlib

def main():
  fdir = '//scepto.caltech.edu/Parker_lab/Christina_Julian/8_10_2018/'
  
  #allfiles = os.listdir(fdir)
  #im_names = [f for f in allfiles if '.tiff' in f]
  #im_names = im_names[0:20]

  #ims = [skimage.io.imread(fdir + f) for f in im_names]
  #im_bg = seg.construct_bg_img(ims=ims, num_ims=20)
  #import matplotlib
  #matplotlib.image.imsave('C:/Users/1004p/Desktop/im_bg.tiff', im_bg)
  #im_bg = skimage.io.imread('C:/Users/1004p/Desktop/Caltech Amgen 2018/Behavioral Analysis/sceptovideo/im_bg.tiff', as_gray = True)
  #im = seg.bg_subtract(ims[0], im_bg)
  
  #a = tracking.process_im(im, props = 'sec', thresh_list=[(200, 9000), (0.85, 1)])
  #print(a)


def rand_ecdf():
  major = int(53.3) #np.mean(test['major_axis_length'])
  minor = int(20.8) #np.mean(test['minor_axis_length'])
  arena = np.ones((1750, 1750))
  rr, cc = skimage.draw.circle(875, 875, 875)
  arena[rr, cc] = 0

  spread = [0 for i in range(100)]
  soc = [0 for i in range(100)]

  for i in range(100):
    im_rand, centroids = sociality.rand_data(arena, 10, major, minor)
    #matplotlib.image.imsave('C:/Users/1004p/Desktop/Caltech Amgen 2018/Behavioral Analysis/im_rand1.tiff', im_rand)
    spread[i] = sociality.clustering(centroids, mult_frame=False)
    soc[i] = sociality.sociality(centroids, mult_frame = False, radius = np.mean([major, minor]))
    #bokeh.io.show(bootcamp_utils.bokeh_imshow(im_rand))
  spreadx, spready = bootcamp_utils.ecdf_vals(np.array(spread), formal=True)
  socx, socy = bootcamp_utils.ecdf_vals(np.array(soc), formal=True)
  formal = alt.Chart([spreadx, spready]
            ).mark_line(
            ).encode(
              x = 'spread:Q',
              y = 'ECDF:Q')
  formal

if __name__ == '__main__':
    rand_ecdf()
    main()