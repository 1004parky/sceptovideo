import re
import segmentation
import numpy as np

def fix_FLIR_file_order(im_files, regex='([^\-]+\-)\d+\-(\d+)(\.tiff)'):
    """Function to sort a list of tiff files from a FLIR SpinView
    run based on their index instead of the default read in order
    from glob.glob.
    
    Parameters
    ----------
    im_files : array like
        a set of image file names from a glob command
    regex : string
        a regular expression used to find the index number for sorting.
        Unless the user named image files with a '-', or saved the file
        as something other than a 'tiff', this should not need to be changed
    Returns
    -------
    sorted_fs : list
        A list of the files in order based on their index number
    """
    
    if not segmentation._check_array_like(im_files):
        raise RuntimeError('im_files must be array like, provided object has type ' + str(type(im_files)))
    if not len(np.array(im_files).shape) == 1:
        raise RuntimeError('The im_files object must be 1D, provided object has shape ' + str(np.array(im_files).shape))
    
    file_dict = {}
    for file in im_files:
        m = re.search(regex, file)
        index = format(int(m.group(2)), '08d')
        file_dict[index] = file
    sorted_fs = [file_dict[ind] for ind in np.sort(list(file_dict.keys()))]
    
    return sorted_fs