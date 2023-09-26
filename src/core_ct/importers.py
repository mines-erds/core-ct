from .core import Core
from pydicom import dcmread
from os import listdir
import os.path
import numpy as np

def dicom(dir: str = None, files: list[str] = None) -> Core:
    """
    Load a DICOM dataset into a 3D numpy array containing brightness levels for each voxel.

    Files containing the DICOM dataset can be specifed by providing a directory or a list of files. If both `dir` and `files` are provided, `dir` will be ignored.

    Note: When specifying a directory all files in that directory will be treated as part of the DICOM dataset. If this is undesirable, use `files` instead.
    """

    # if files was not provided, load files from the provided directory
    if files == None:
        # throw error if directory not provided
        if dir == None:
            raise Exception("Must provide a directory or file list")
        # get the list of files for the core
        files = [os.path.join(dir, file_name) for file_name in listdir(dir)]

    # skip files with no SliceLocation information (should be a float)
    slices = []
    skipped: list[str] = []
    for f in files:
        ds = dcmread(f)
        if isinstance(ds.SliceLocation, float):
            slices.append(ds)
        else:
            skipped.append(f)

    if len(skipped) > 0:
        raise Exception(f"Failed to load {len(skipped)} files, missing SliceLocation: {skipped}")

    # re-sort to put the slices in the right order
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    # pixel dimensions, assuming all slices are the same
    ps = slices[0].PixelSpacing
    ss = slices[0].SliceThickness

    # create 3D array
    img_shape: list = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d: np.ndarray[np.float64] = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d
    
    return Core(pixel_array=img3d, pixel_dimensions=[ps, ps, ss])