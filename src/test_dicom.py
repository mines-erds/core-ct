from importers import dicom
from glob import glob
from PIL import Image as im
import numpy as np
import argparse

from core import Core

def main() -> None:
    parser = argparse.ArgumentParser(description="Test loading of a dicom dataset")
    parser.add_argument("dir", metavar="DIRECTORY", type=str, help="Path to the directory containing dicom dataset")

    args = parser.parse_args()
    dir: str = args.dir

    dicom_from_dir(dir)
    glob_str = dir + "/*"
    dicom_from_files(glob(glob_str))

def dicom_from_dir(dir: str) -> None:
    core: Core = dicom(dir=dir)
    display_core(core, "dicom_from_dir.png", axis=2, index=0)

def dicom_from_files(files: list[str]) -> None:
    core: Core = dicom(files=files)
    display_core(core, "dicom_from_files.png", axis=2, index=0)

def display_core(core: Core, output: str, axis: int = 2, index: int = 0) -> None:
    """
    Output an image containing the slice at the provided z index in the dicom dataset
    """
    # retrieve slice data
    match axis:
        case 0:
            slice = core.pixel_array[index]
        case 1:
            slice = core.pixel_array[:,index]
        case 2:
            slice = core.pixel_array[:,:,index]
        case _:
            raise Exception("axis must be a value between 0 and 2 (inclusive)")

    # find the min and max brightness value to help with normalizing
    max: float = np.max(slice)
    min: float = np.min(slice)

    # iterate through every datapoint in the slice and normalize it to be an integer between 0 and 255 (inclusive)
    offset = abs(min)
    norm_max = max + offset
    for i in range(len(slice)):
        for j in range(len(slice[0])):
            slice[i][j] = int(((slice[i][j] + offset) / norm_max) * 255)

    # create an image depicting the slice using our normalized values
    picture = im.fromarray(slice)
    picture = picture.convert("L")
    picture.save(output)

if __name__ == "__main__":
    main()