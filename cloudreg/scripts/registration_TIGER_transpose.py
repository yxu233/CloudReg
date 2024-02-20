#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:57:24 2024

@author: user
"""


import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

import math

def plot_max(im, ax=0):
    
    max_im = np.amax(im, axis=0)
    plt.figure(); plt.imshow(max_im)
    


def imgResample(img, spacing, size=[], useNearest=False, origin=None, outsideValue=0):
    """Resample image to certain spacing and size.

    Args:
        img (SimpleITK.Image): Input 3D image.
        spacing (list): List of length 3 indicating the voxel spacing as [x, y, z]
        size (list, optional): List of length 3 indicating the number of voxels per dim [x, y, z] (the default is [], which will use compute the appropriate size based on the spacing.)
        useNearest (bool, optional): If True use nearest neighbor interpolation. (the default is False, which will use linear interpolation.)
        origin (list, optional): The location in physical space representing the [0,0,0] voxel in the input image. (the default is [0,0,0])
        outsideValue (int, optional): value used to pad are outside image (the default is 0)

    Returns:
        SimpleITK.Image: Resampled input image.
    """

    if origin is None:
        origin = [0] * 3
    if len(spacing) != img.GetDimension():
        raise Exception("len(spacing) != " + str(img.GetDimension()))

    # Set Size
    if size == []:
        inSpacing = img.GetSpacing()
        inSize = img.GetSize()
        size = [
            int(math.ceil(inSize[i] * (inSpacing[i] / spacing[i])))
            for i in range(img.GetDimension())
        ]
    else:
        if len(size) != img.GetDimension():
            raise Exception("len(size) != " + str(img.GetDimension()))

    # Resample input image
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
    identityTransform = sitk.Transform()

    return sitk.Resample(
        img,
        size,
        identityTransform,
        interpolator,
        origin,
        spacing,
        img.GetDirection(),
        outsideValue,
    )



# im = tiff.imread('/media/user/FantomHD/Test_registration/Test_registration_stripe_filt_registration_2ndtry_5000iterations/downloop_1_target_to_labels_highres.tif')

# im = np.moveaxis(im, 0, 1)

# im = np.rot90(im, k=-1, axes=(1, 2))

# im = np.flip(im, axis=0)

# plot_max(im)

# #tiff.imwrite('/media/user/FantomHD/Test_registration/Test_registration_stripe_filt_registration_2ndtry_5000iterations/downloop_1_target_to_labels_highres.tif' + '_ROTATED' , im)
# tiff.imwrite('/home/user/CloudReg_tests_stripe_filt_registration_atlas_30um_1e4_regularize/downloop_1_target_to_labels_highres.tif' + '_ROTATED' , im)
# tiff.imwrite('/media/user/FantomHD/Test_registration/Test_registration_stripe_filt_registration_2ndtry_5000iterations/downloop_1_target_to_labels_highres.tif' + '_ROTATED' , im)



#%%
### FOR CONVERTING THE PERENS ATLAS

from skimage.transform import rescale, resize, downscale_local_mean

#im = tiff.imread('/home/user/.brainglobe/perens_lsfm_mouse_20um_v1.0/annotation.tiff')

im = tiff.imread('/home/user/.brainglobe/perens_lsfm_mouse_20um_v1.0/reference.tiff')

save_name = '/home/user/.brainglobe/perens_lsfm_mouse_20um_v1.0/reference_FOR_CLOUDREG.nrrd'

#atlas_path = '/home/user/.brainglobe/perens_lsfm_mouse_20um_v1.0/'
atlas_path = '/home/user/.brainglobe/princeton_mouse_20um_v1.0/'


#im = tiff.imread('/home/user/.brainglobe/perens_lsfm_mouse_20um_v1.0/annotation.tiff')

im = tiff.imread(atlas_path + 'reference.tiff')

save_name = atlas_path + 'atlas_data.nrrd'
save_100 = atlas_path + 'average_template_100.nrrd'

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


###################### FAKE THE RESOLUTION BECAUSE THIS ATLAS IS SHRUNKEN!!!
orig_res = 25



#new_res = 40   ### THIS WORKS WELL SO FAR

new_res = 40



import SimpleITK as sitk
img_s = sitk.GetImageFromArray(img)


# set spacing in microns
resolution = [orig_res, orig_res, orig_res]
img_s.SetSpacing(resolution)
img_s = imgResample(img_s, [new_res, new_res, new_res])
sitk.WriteImage(img_s, save_name)


### ALSO write low res 100um volume for warmstart
img_s = imgResample(img_s, [100, 100, 100])
sitk.WriteImage(img_s, save_100)



#%%
###################### FOR THE ANNOTATIONS

im = tiff.imread(atlas_path + 'annotation.tiff')

save_name = atlas_path + 'parcellation_data.nrrd'
save_100 = atlas_path +'annotation_100.nrrd'

save_tiff = atlas_path + 'parcellation_data.tif'


im = np.moveaxis(im, 2, 0)
im = np.flip(im, axis=1)
img = np.rot90(im, k=-1, axes=(1, 2))

#resolution = 20 * 1000

### SAVE HIGH RES AS TIFF

import SimpleITK as sitk
img_s = sitk.GetImageFromArray(img)
# set spacing in microns
#resolution = np.divide(resolution, 1000.0).tolist()


#resolution = [20, 20, 20]

resolution = [orig_res, orig_res, orig_res]
img_s.SetSpacing(resolution)

import tifffile as tiff
tiff.imwrite(save_tiff, sitk.GetArrayFromImage(img_s))
     


### SAVE DOWNSAMPLED   

im_rescaled = rescale(img, scale_diff)
img_s = sitk.GetImageFromArray(img)
resolution = [new_res, new_res, new_res]   ### rescale
#img_s.SetSpacing(resolution)
#sitk.WriteImage(img_s, save_name)


img_s = imgResample(img_s, [new_res, new_res, new_res])
sitk.WriteImage(img_s, save_name)

### ALSO write low res 100um volume for warmstart
img_s = imgResample(img_s, [100, 100, 100])
sitk.WriteImage(img_s, save_100)


