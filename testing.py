import skimage.io
import scipy.spatial.distance
import pandas as pd

import tracking


def main():
  im_o = skimage.io.imread('toy_data/ant_scepto-04092018133745-6089.tiff')
  im_n = skimage.io.imread('toy_data/ant_scepto-04092018133745-6090.tiff')
  #oim = tracking.process_im(im_o, thresh_list=[(200, 1200), (0.9, 1)])
  nim = tracking.process_im(im_n, props = 'sec',thresh_list=[(200, 2000), (0.9, 1)])

  #print(oim)
  print(nim)

  #data = pd.DataFrame(columns=[0, 1, 2, 3, 4])

  #a, g = tracking.match_blobs(oim, nim)
  #print(a,g)
  print(tracking.track('toy_data/ant_scepto-04092018133745-6089.tiff', thresh_list=[(200, 1200), (0.9, 1)]))

if __name__ == '__main__':
    main()