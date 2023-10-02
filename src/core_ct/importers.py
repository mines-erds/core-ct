from .core import Core
from pydicom import dcmread
from os import listdir
import os.path
import numpy as np
import re

def dicom(dir: str = None, files: list[str] = None, force: bool = False, ignore_hidden_files: bool = True) -> Core:
    """
    Load a DICOM dataset into a `Core` object containing brightness values, voxel dimensions, etc.

    Files containing the DICOM dataset can be specifed by providing a directory or a list of files. If both `dir` and `files` are provided, `dir` will be ignored.

    Note: when specifying a directory all files in that directory will be treated as part of the DICOM dataset. If this is undesirable, use `files` instead.

    ## Parameters
        dir: path to directory containing DICOM dataset (ignored if `files` is specified)
        files: list of filepaths belonging to DICOM dataset
        force: if set to `True`, files that produce errors during reading will be ignored
        ignore_hidden_files: if set to `True`, hidden files (names starting with ".") will be ignored
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
        # check if we should ignore this file
        # this regex tries to account for which slash type the system uses ("/" in Linux/Unix/MacOS, "\" in Windows)
        hidden_file_regex: str = r"([\\/])\.[^\1]+$"
        if ignore_hidden_files and re.search(hidden_file_regex, f):
            continue

        # try to read slice
        try:
            ds = dcmread(f, force=force)
        except:
            if not force:
                # forward pydicom exception so the stack trace is more useful
                raise
            else:
                continue
        
        # make sure SliceLocation exists in the slice
        try:
            if isinstance(ds.SliceLocation, float):
                slices.append(ds)
            else:
                skipped.append(f)
        # in case SliceLocation isn't an attribute of ds
        except Exception as pydicom_exception:
            if not force:
                raise Exception(f"File does not contain SliceLocation in header: {f}") from pydicom_exception
            else:
                skipped.append(f)

    if not force and len(skipped) > 0:
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