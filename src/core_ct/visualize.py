"""Methods that assist in visualizing the CT scans of rock cores."""

import numpy as np
from core_ct.core import Core
from PIL import Image as im


def slice(core: Core, output: str, axis: int = 2, index: int = 0) -> None:
    """
    Output an image containing the slice at the provided index.

    Arguments:
    ---------
        core: the `Core` object to take the slice of
        output: the name of the outputted image file
        axis: which axis to take a slice of
        index: where on the axis to take the slice of
    """
    # retrieve slice data
    match axis:
        case 0:
            slice = core.pixel_array[index]
        case 1:
            slice = core.pixel_array[:, index]
        case 2:
            slice = core.pixel_array[:, :, index]
        case _:
            raise Exception("axis must be a value between 0 and 2 (inclusive)")

    # find the min and max brightness value to help with normalizing
    max: float = np.max(slice)
    min: float = np.min(slice)

    # normalize each data point to be an integer between 0 and 255 (inclusive)
    offset = abs(min)
    norm_max = max + offset
    for i in range(len(slice)):
        for j in range(len(slice[0])):
            slice[i][j] = int(((slice[i][j] + offset) / norm_max) * 255)

    # create an image depicting the slice using our normalized values
    picture = im.fromarray(slice)
    picture = picture.convert("L")
    picture.save(output)
