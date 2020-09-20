# local imports
from util import get_reorientations, aws_cli
from visualization import (
    ara_average_data_link,
    ara_annotation_data_link,
    create_viz_link,
    S3Url,
)
from download_data import download_data
from ingest_image_stack import ingest_image_stack

import shlex
from cloudvolume import CloudVolume
from scipy.spatial.transform import Rotation
import numpy as np
import argparse
import subprocess
import os

atlas_orientation = "PIR"


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
):
    """Run EM-LDDMM registration on precomputed volume at input_s3_path

    Args:
        input_s3_path (str): S3 path to precomputed data to be registered
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
    """

    # get volume info
    s3_url = S3Url(input_s3_path)
    channel = s3_url.key.split("/")[-1]
    exp = s3_url.key.split("/")[-2]

    # only after stitching autofluorescence channel
    base_path = os.path.expanduser("~/")
    registration_prefix = f"{base_path}/{exp}_{channel}_registration/"
    target_name = f"{base_path}/autofluorescence_data.tif"

    # download downsampled autofluorescence channel
    print("downloading data for registration...")
    voxel_size = download_data(input_s3_path, target_name)
    # if high res atlas labels file doesn't exist
    ara_annotation_10um = os.path.expanduser(
        "~/CloudReg/registration/atlases/ara_annotation_10um.tif"
    )
    if not os.path.exists(ara_annotation_10um):
        # download it
        _ = download_data(
            ara_annotation_data_link(10), ara_annotation_10um, desired_resolution=10000
        )

    # initialize affine transformation for data
    atlas_res = 100
    atlas_s3_path = ara_average_data_link(atlas_res)
    initial_affine = get_affine_matrix(
        translation,
        rotation,
        atlas_orientation,
        orientation,
        fixed_scale,
        atlas_s3_path,
    )

    # run registration
    affine_string = [", ".join(map(str, i)) for i in initial_affine]
    affine_string = "; ".join(affine_string)
    matlab_registration_command = f"""
        matlab -nodisplay -nosplash -nodesktop -r \"niter={num_iterations};sigmaR={regularization};missing_data_correction={int(missing_data_correction)};grid_correction={int(grid_correction)};bias_correction={int(bias_correction)};base_path=\'{base_path}\';target_name=\'{target_name}\';registration_prefix=\'{registration_prefix}\';dxJ0={voxel_size};fixed_scale={fixed_scale};initial_affine=[{affine_string}];run(\'~/CloudReg/registration/registration_script_mouse_GN.m\'); exit;\"
    """
    print(matlab_registration_command)
    subprocess.run(shlex.split(matlab_registration_command))

    # save results to S3
    if log_s3_path:
        # sync registration results to log_s3_path
        aws_cli(["s3", "sync", registration_prefix, log_s3_path])

    # upload high res deformed atlas and deformed target to S3
    ingest_image_stack(
        output_s3_path,
        voxel_size,
        f"{registration_prefix}/downloop_2_labels_to_target_highres.img",
        "img",
        "uint64",
    )

    # print out viz link for visualization
    # visualize results at 5 microns
    viz_link = create_viz_link(
        [input_s3_path, output_s3_path], output_resolution=np.array([5] * 3) / 1e6
    )
    print("###################")
    print(f"VIZ LINK: {viz_link}")
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

    # affine initialization args
    parser.add_argument(
        "-orientation", help="3-letter orientation of data. i.e. LPS", type=str
    )
    parser.add_argument(
        "--fixed_scale",
        help="Fixed scale of data, uniform in all dimensions. Default is 1.",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0]
    )
    parser.add_argument(
        "--translation",
        help="Initial translation in x,y,z respectively in microns.",
        nargs="+",
        type=float,
        default=[0, 0, 0],
    )
    parser.add_argument(
        "--rotation",
        help="Initial rotation in x,y,z respectively in degrees.",
        nargs="+",
        type=float,
        default=[0, 0, 0],
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
    )

    # registration params
    parser.add_argument(
        "--regularization",
        help="Weight of the regularization. Bigger regularization means less regularization. Default is 5e3",
        type=float,
        default=5e3,
    )
    parser.add_argument(
        "--iterations",
        help="Number of iterations to do at low resolution. Default is 5000.",
        type=int,
        default=3000,
    )

    args = parser.parse_args()

    register(
        args.input_s3_path,
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
    )
