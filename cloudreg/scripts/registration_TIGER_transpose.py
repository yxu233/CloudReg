#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:57:24 2024

@author: user
"""


import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt


def plot_max(im, ax=0):
    
    max_im = np.amax(im, axis=0)
    plt.figure(); plt.imshow(max_im)
    

# im = tiff.imread('/home/user/CloudReg_tests_stripe_filt_registration_atlas_30um_1e4_regularize/downloop_1_target_to_labels_highres.tif')


# im = np.moveaxis(im, 0, 1)

# im = np.rot90(im, k=-1, axes=(1, 2))

# im = np.flip(im, axis=0)

# plot_max(im)

# #tiff.imwrite('/media/user/FantomHD/Test_registration/Test_registration_stripe_filt_registration_2ndtry_5000iterations/downloop_1_target_to_labels_highres.tif' + '_ROTATED' , im)
# tiff.imwrite('/home/user/CloudReg_tests_stripe_filt_registration_atlas_30um_1e4_regularize/downloop_1_target_to_labels_highres.tif' + '_ROTATED' , im)



#%%
### FOR CONVERTING THE PERENS ATLAS

from skimage.transform import rescale, resize, downscale_local_mean

#im = tiff.imread('/home/user/.brainglobe/perens_lsfm_mouse_20um_v1.0/annotation.tiff')

im = tiff.imread('/home/user/.brainglobe/perens_lsfm_mouse_20um_v1.0/reference.tiff')

save_name = '/home/user/.brainglobe/perens_lsfm_mouse_20um_v1.0/reference_FOR_CLOUDREG.nrrd'

im = np.moveaxis(im, 2, 0)
im = np.flip(im, axis=1)
img = np.rot90(im, k=-1, axes=(1, 2))

#resolution = 20 * 1000
orig_res = 20
new_res = 30

scale_diff = orig_res/new_res

im_rescaled = rescale(img, scale_diff)  ### rescale

import SimpleITK as sitk
img_s = sitk.GetImageFromArray(img)
# set spacing in microns
resolution = [new_res, new_res, new_res]
img_s.SetSpacing(resolution)
sitk.WriteImage(img_s, save_name)



###################### FOR THE ANNOTATIONS

im = tiff.imread('/home/user/.brainglobe/perens_lsfm_mouse_20um_v1.0/annotation.tiff')

save_name = '/home/user/.brainglobe/perens_lsfm_mouse_20um_v1.0/annotation_FOR_CLOUDREG_30um.nrrd'
save_tiff = '/home/user/.brainglobe/perens_lsfm_mouse_20um_v1.0/annotation_FOR_CLOUDREG_20um.tiff'

im = np.moveaxis(im, 2, 0)
im = np.flip(im, axis=1)
img = np.rot90(im, k=-1, axes=(1, 2))

#resolution = 20 * 1000

### SAVE HIGH RES AS TIFF

import SimpleITK as sitk
img_s = sitk.GetImageFromArray(img)
# set spacing in microns
#resolution = np.divide(resolution, 1000.0).tolist()

resolution = [20, 20, 20]
img_s.SetSpacing(resolution)

import tifffile as tiff
tiff.imwrite(save_tiff, sitk.GetArrayFromImage(img_s))
     


### SAVE DOWNSAMPLED   
im_rescaled = rescale(img, scale_diff)
img_s = sitk.GetImageFromArray(img)
resolution = [new_res, new_res, new_res]   ### rescale
img_s.SetSpacing(resolution)
sitk.WriteImage(img_s, save_name)












