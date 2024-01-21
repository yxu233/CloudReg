# local imports
from .util import get_reorientations, aws_cli
from .visualization import (
    ara_average_data_link,
    ara_annotation_data_link,
    create_viz_link,
    S3Url,
)
from .download_data import download_data
from .ingest_image_stack import ingest_image_stack

import shlex
from cloudvolume import CloudVolume
from scipy.spatial.transform import Rotation
import numpy as np
import argparse
import subprocess
import os


def get_affine_matrix(
    translation,
    rotation,
    from_orientation,
    to_orientation,
    fixed_scale,
    s3_path,
    center=False,
):
    """Get Neuroglancer-compatible affine matrix transfrming precomputed volume given set of translations and rotations

    Args:
        translation (list of float): x,y,z translations respectively in microns
        rotation (list of float): x,y,z rotations respectively in degrees
        from_orientation (str): 3-letter orientation of source data 
        to_orientation (str): 3-letter orientation of target data
        fixed_scale (float): Isotropic scale factor
        s3_path (str): S3 path to precomputed volume for source data
        center (bool, optional): If true, center image at it's origin. Defaults to False.

    Returns:
        np.ndarray: Returns 4x4 affine matrix representing the given translations and rotations of source data at S3 path
    """

    # since neuroglancer uses corner 0 coordinates we need to center the volume at it's center
    vol = CloudVolume(s3_path)
    # volume size in um
    vol_size = np.multiply(vol.scales[0]["size"], vol.scales[0]["resolution"]) / 1e3
    # make affine matrix in homogenous coordinates
    affine = np.zeros((4, 4))
    affine[-1, -1] = 1
    order, flips = get_reorientations(from_orientation, to_orientation)
    # reorder vol_size to match reorientation
    vol_size = vol_size[order]
    dim = affine.shape[0]
    # swap atlas axes to match target
    affine[range(len(order)), order] = 1
    # flip across appropriate dimensions
    affine[:3, :3] = np.diag(flips) @ affine[:3, :3]

    if center:
        # for each flip add the size of image in that dimension
        affine[:3, -1] += np.array(
            [vol_size[i] if flips[i] == -1 else 0 for i in range(len(flips))]
        )
        # make image centered at the middle of the image
        # volume is now centered
        affine[:3, -1] -= vol_size / 2

    # get rotation matrix
    if np.array(rotation).any():
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = Rotation.from_euler(
            "xyz", rotation, degrees=True
        ).as_matrix()
        # compose rotation with affine
        affine = rotation_matrix @ affine
    # add translation components
    # note: for neuroglancer affine, we scale the translations by voxel size
    # because neuroglancer expects translation in voxels
    affine[:3, -1] += translation

    # scale by fixed_scale
    if isinstance(fixed_scale, float):
        affine = np.diag([fixed_scale, fixed_scale, fixed_scale, 1.0]) @ affine
    elif isinstance(fixed_scale, (list, np.ndarray)) and len(fixed_scale) == 3:
        affine = np.diag([fixed_scale[0], fixed_scale[1], fixed_scale[2], 1.0]) @ affine 
    else:
        affine = np.diag([fixed_scale[0], fixed_scale[0], fixed_scale[0], 1.0]) @ affine

    return affine


def register(
    input_s3_path,
    atlas_s3_path,
    parcellation_s3_path,
    atlas_orientation,
    output_s3_path,
    log_s3_path,
    orientation,
    fixed_scale,
    translation,
    rotation,
    missing_data_correction,
    grid_correction,
    bias_correction,
    regularization,
    num_iterations,
    registration_resolution
):
    """Run EM-LDDMM registration on precomputed volume at input_s3_path

    Args:
        input_s3_path (str): S3 path to precomputed data to be registered
        atlas_s3_path (str): S3 path to atlas to register to.
        parcellation_s3_path (str): S3 path to atlas to register to.
        atlas_orientation (str): 3-letter orientation of atlas
        output_s3_path (str): S3 path to store precomputed volume of atlas transformed to input data
        log_s3_path (str): S3 path to store intermediates at
        orientation (str): 3-letter orientation of input data
        fixed_scale (float): Isotropic scale factor on input data
        translation (list of float): Initial translations in x,y,z of input data
        rotation (list): Initial rotation in x,y,z for input data
        missing_data_correction (bool): Perform missing data correction to ignore zeros in image
        grid_correction (bool): Perform grid correction (for COLM data)
        bias_correction (bool): Perform illumination correction
        regularization (float): Regularization constat in cost function. Higher regularization constant means less regularization
        num_iterations (int): Number of iterations of EM-LDDMM to run
        registration_resolution (int): Minimum resolution at which the registration is run.
    """

    # get volume info
    s3_url = S3Url(input_s3_path)
    channel = s3_url.key.split("/")[-1]
    exp = s3_url.key.split("/")[-2]

    # only after stitching autofluorescence channel
    base_path = os.path.expanduser("~/")
    registration_prefix = f"{base_path}/{exp}_{channel}_registration_atlas_25um/"  ### changes name of output folder
    atlas_prefix = f'{base_path}/CloudReg/cloudreg/registration/atlases/'
    target_name = f"{base_path}/autofluorescence_data.tif"
    atlas_name = f"{atlas_prefix}/atlas_data.nrrd"
    parcellation_name = f"{atlas_prefix}/parcellation_data.nrrd"
    parcellation_hr_name = f"{atlas_prefix}/parcellation_data.tif"

    # download downsampled autofluorescence channel
    print("downloading input data for registration... YO")
    
    print('Registration minimum resolution is: ' + str(registration_resolution))
    
    
    # convert to nanometers
    registration_resolution *= 1000.0 
    
   
    

    
    #fixed_scale = [1.0, 1.0, 0.75]   ### current run is with this
    
    
    
    
    
    # download raw data at lowest 15 microns  15000, low res 30000
    # voxel_size = download_data(input_s3_path, target_name, resample_isotropic=True, desired_resolution=20000)
    
    #voxel_size = [23.590400000000002, 23.590400000000002, 4.8]

    #voxel_size = [9.22, 9.22, 10]
    
    #voxel_size = [18.44, 18.44, 10]
    
    """ Tiger skipping this download
    
            ***WILL NOT WRITE A NEW AUTOFLUORESCENCE FILE!!!
    """
    print('SKIPPING DOWNLOAD OF AUTOFLUORESCENCE')
    #voxel_size = [9.216, 9.216, 6.0]   ### for cuprizone data
    
    
    #voxel_size = [8.0, 8.0, 5.0]     ### for small 5x volume 8bit
    #voxel_size = [12.0, 12.0, 3.75]  ### after rescaling to 75% of original  ### for small 5x volume 8bit
    
    
    #voxel_size = [11.52, 11.52, 5.0]  ### after rescaling for 5x LARGER volume
    #voxel_size = [9.6, 9.6, 5.0]
    #voxel_size = [8.0, 8.0, 5.0]

    
    #voxel_size = [11.04, 11.04, 9.0]
    voxel_size=[20, 20, 20]
    
    
        
    
    
    """ Write function to pad the autofluorescence data
    
            how come this is changing the size of the atlases???
    """
    print('RESCALING AUTOFLUORESCENT DATA')
    # import tifffile as tiff
    # #target_name = 'autofluorescence_data.tif'
    # im = tiff.imread(target_name)
    
    
    # dim = im.shape
    
    #voxel_size = np.moveaxis(np.asarray(voxel_size), -1, 0)
    
    
    ### make lateral dimension (waist) at least 14mm
    # x_p = y_p = z_p = 0;
    # y_min = 14000
    # if dim[2] * voxel_size[0] <= y_min:
    #     missing = y_min - (dim[2] * voxel_size[0])
    #     y_p = int((missing/2)/ voxel_size[0])  ### scale back to num voxels
    
    
    # ### make vertical dimension (top to bottom) at least 17 mm
    # x_min = 17000
    # if dim[1] * voxel_size[1] <= x_min:
    #     missing = x_min - (dim[1] * voxel_size[1])
    #     x_p = int((missing/2)/ voxel_size[1])  ### scale back to num voxels
    


    # ### make Z dimension at least 12 mm
    # z_min = 10000
    # if dim[0] * voxel_size[-1] <= z_min:
        
    #     missing = z_min - (dim[0] * voxel_size[-1])
    #     z_p = int((missing/2)/ voxel_size[-1])  ### scale back to num voxels
    
    
    
    # width = ((z_p, z_p), (x_p, x_p), (y_p, y_p))
    # im_pad = np.pad(im, width, constant_values = 0)
 
    
    """ ### MASK OUT PORTIONS OF TISSUE """
    # im_pad[:, 0:202, :] = 0
    # im_pad[:, 1263:-1, :] = 0
    
    #im_pad[:, 0:392, :] = 0
    #im_pad[:, 1180:-1, :] = 0    
    
    
    ### take out right hemisphere
    #im_pad[:, :, 665:-1] = 0
    
 
    #tiff.imsave(target_name, im_pad)
    
    
    
    
    

    
    
    

    # download atlas and parcellations at registration resolution
    print('First download')
    print(voxel_size)
    
    
    
    ### HACK - Tiger, voxel_size needs to be a numpy array so can actually multiply later, or else makes a 1000 entry array during multiplication...
    voxel_size = np.asarray(voxel_size)
    
    
    
    
    
    
    
    
    
    
    """ Tiger hack, do I need these every time??? """
    print('SKIPPING DOWNLOAD OF ATLAS')
    # atlas_vox_size = download_data(atlas_s3_path, atlas_name, registration_resolution, resample_isotropic=True)
    # print('downloaded atlas')
    # print(atlas_vox_size)

    
    
    # parcel_vox_size = download_data(parcellation_s3_path, parcellation_name, registration_resolution, resample_isotropic=True)
    # print('downloaded parcel')
    # print(parcel_vox_size)    

    
    
    # also download high resolution parcellations for final transformation
    microns_min = 20000
    # parcellation_voxel_size, parcellation_image_size = download_data(parcellation_s3_path, parcellation_hr_name, microns_min, resample_isotropic=True, return_size=True)

    
    print('parcellation voxel size')
    ### for microns_min = 20000
    parcellation_voxel_size = [20.0, 20.0, 20.0]
    #parcellation_image_size = [660, 400, 1140]
    parcellation_image_size = [660, 400, 570] ### SINCE WE DOWNSAMPLED???


    print(parcellation_voxel_size)
    print(parcellation_image_size)



    print(orientation)
    print(atlas_orientation)

    
    
    # initialize affine transformation for data
    initial_affine = get_affine_matrix(
        translation,
        rotation,
        atlas_orientation,
        orientation,
        fixed_scale,
        atlas_s3_path,
        #center=True
    )
    
    
    ### red --> first axis, Posterior, then green --> second axis, Inferior, then right --> Right (PIR) overall allen
    print(initial_affine)
    

    
    # this is the initialization for registration
    target_affine = get_affine_matrix(
        [0] * 3, [0] * 3, orientation, orientation, 1.0, input_s3_path)#, center=True)

    # get viz link from input link
    viz_link = create_viz_link(
        [input_s3_path, atlas_s3_path],
        affine_matrices=[target_affine, initial_affine],
    )

    # ask user if this initialization looks right
    # user_input = ""
    # while user_input == "":
    #     user_input = input(f"Does this initialization look right? {viz_link} (y/n): ")
    # # if no quit and ask for another initialization
    # if user_input == "n":
    #     raise (Exception("Please rerun with new initialization"))
        
        
        
        


    # run registration
    affine_string = [", ".join(map(str, i)) for i in initial_affine]
    affine_string = "; ".join(affine_string)
    print(affine_string)
    print('bias_correction is: ' + str(bias_correction))
    matlab_registration_command = f"""
        matlab -nodisplay -nosplash -nodesktop -r \"niter={num_iterations};sigmaR={regularization};missing_data_correction={int(missing_data_correction)};grid_correction={int(grid_correction)};bias_correction={int(bias_correction)};base_path=\'{base_path}\';target_name=\'{target_name}\';registration_prefix=\'{registration_prefix}\';atlas_prefix=\'{atlas_prefix}\';dxJ0={voxel_size};fixed_scale={fixed_scale};initial_affine=[{affine_string}];parcellation_voxel_size={parcellation_voxel_size};parcellation_image_size={parcellation_image_size};run(\'~/CloudReg/cloudreg/registration/map_nonuniform_multiscale_v02_mouse_gauss_newton.m\'); exit;\"
    """
    #-nodisplay -nosplash -nodesktop
    
    print(matlab_registration_command)
    subprocess.run(shlex.split(matlab_registration_command))

    # save results to S3
    # if log_s3_path:
    #     # sync registration results to log_s3_path
    #     aws_cli(["s3", "sync", registration_prefix, log_s3_path])

    # upload high res deformed atlas and deformed target to S3
    ingest_image_stack(
        output_s3_path,
        voxel_size*1000,
        f"{registration_prefix}/downloop_1_labels_to_target_highres.img",
        "img",
        "uint64",
    )

    # print out viz link for visualization
    # visualize results at 5 microns
    viz_link = create_viz_link(
        [input_s3_path, output_s3_path], output_resolution=np.array([5] * 3) / 1e6
    )
    print("###################")
    print(f"VIZ LINK (atlas overlayed on data): {viz_link}")
    print("###################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run COLM pipeline on remote EC2 instance with given input parameters"
    )
    # data args
    parser.add_argument(
        "-input_s3_path",
        help="S3 path to precomputed volume used to register the data",
        type=str,
    )
    parser.add_argument(
        "-log_s3_path",
        help="S3 path at which registration outputs are stored.",
        type=str,
    )
    parser.add_argument(
        "--output_s3_path",
        help="S3 path to store atlas transformed to target as precomputed volume. Should be of the form s3://<bucket>/<path_to_precomputed>. Default is same as input s3_path with atlas_to_target as channel name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--atlas_s3_path",
        help="S3 path to atlas we want to register to. Should be of the form s3://<bucket>/<path_to_precomputed>. Default is Allen Reference atlas path",
        type=str,
        default=ara_average_data_link(100),
    )
    parser.add_argument(
        "--parcellation_s3_path",
        help="S3 path to corresponding atlas parcellations. If atlas path is provided, this should also be provided. Should be of the form s3://<bucket>/<path_to_precomputed>. Default is Allen Reference atlas parcellations path",
        type=str,
        default=ara_annotation_data_link(10),
    )
    parser.add_argument(
        "--atlas_orientation",
        help="3-letter orientation of data. i.e. LPS",
        type=str,
        default='PIR'
        #default='PSR'
    )

    # affine initialization args
    parser.add_argument(
        "-orientation", help="3-letter orientation of data. i.e. LPS", type=str
    )
    parser.add_argument(
        "--fixed_scale",
        help="Fixed scale of data, uniform in all dimensions. Default is 1.",
        nargs='+',
        type=float,
        default=[1.0, 1.0, 1.0]
    )
    parser.add_argument(
        "--translation",
        help="Initial translation in x,y,z respectively in microns.",
        nargs="+",
        type=float,
        #default=[0, -400, -520],     ### for small 5x volume 8bit
        #default=[-300, 2500, -300],      ### for LARGER volume 5x
        #default=[0, -1500, -1200],              ### for ZOOMED OUT
        #default=[-600, -1200, 5200]
        #default=[-400, -1700, 5600]             ### for ZOOMED OUT BLANK
        #default=[30,7500,11000],    ### for BRAIN to atlas
        
        
        #default=[-500, 2500, 6500], ### for autofluorescence
        default = [0, 0, 0]
        
        #default=[0, 3000, 6000],   ### for ROTATED volume
        
        
            # X moves it left/right
            # Y moves it +Y (is down!!!)
    )
    parser.add_argument(
        "--rotation",
        help="Initial rotation in x,y,z respectively in degrees.",
        nargs="+",
        type=float,
        #default=[5, 0, -8],         ### for small 5x volume 8bit
        #default=[4, 0, -8],           ### for LARGER volume 5x
        #default=[12, 0, 0],           ### for ZOOMED OUT BLANK
        
        #default=[0,0,0],    ### for BRAIN to atlas
        
        default=[0, 0, 0], ### for autofluorescence
        
                ### rotation in X tilts it forward/backward
                ### rotation in Z rotates it left/right clockwise
    )

    # preprocessing args
    parser.add_argument(
        "--bias_correction",
        help="Perform bias correction prior to registration.",
        type=eval,
        choices=[True, False],
        default='True',
    )
    parser.add_argument(
        "--missing_data_correction",
        help="Perform missing data correction by ignoring 0 values in image prior to registration.",
        type=eval,
        choices=[True, False],
        default='True',
    )
    parser.add_argument(
        "--grid_correction",
        help="Perform correction for low-intensity grid artifact (COLM data)",
        type=eval,
        choices=[True, False],
        default='False',
        
        #default='True',
    )

    # registration params
    parser.add_argument(
        "--regularization",
        help="Weight of the regularization. Bigger regularization means less regularization. Default is 5e3",
        type=float,
        default=5e3,
        #default=5e4  ### wasn't great at 5e1, was slightly worse at 5e4 but also at 100 microns doing tests so not accurate at all
    )
    parser.add_argument(
        "--iterations",
        help="Number of iterations to do at low resolution. Default is 5000.",
        type=int,
        #default=5000,
        #default=3000,   ### used this before
        #default=1700,
        #default=1000,
        #default=5000
        #default=5,    ### for testing
        default=500,
        
    )
    parser.add_argument(
        "--registration_resolution",
        help="Minimum resolution that the registration is run at (in microns). Default is 100.",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    register(
        args.input_s3_path,
        args.atlas_s3_path,
        args.parcellation_s3_path,
        args.atlas_orientation,
        args.output_s3_path,
        args.log_s3_path,
        args.orientation,
        args.fixed_scale,
        args.translation,
        args.rotation,
        args.missing_data_correction,
        args.grid_correction,
        args.bias_correction,
        args.regularization,
        args.iterations,
        args.registration_resolution
    )
