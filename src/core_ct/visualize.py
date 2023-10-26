"""Methods that assist in visualizing the CT scans of rock cores."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.colors import Normalize
from core_ct.core import Core
from PIL import Image as im
from .analysis import brightness_trace

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

def display_core(core: Core) -> plt.Axes:
    core_dim = core.pixel_array.shape
    fig, axes = plt.subplots(nrows = 1, ncols = 3, width_ratios=[core_dim[2], core_dim[2], core_dim[1]], figsize = (8,5))
    # fig, axes = plt.subplots(nrows = 1, ncols = 3)
    for i, ax in enumerate(axes):
        slice_loc = core_dim[i]//2
        ax.imshow(core.slice(axis = i, loc = slice_loc))
        ax.set_title("Axis {} collapsed".format(i, slice_loc))
    axes[0].set_ylabel("Axis 1")
    axes[0].set_xlabel("Axis 2")
    axes[1].set_ylabel("Axis 0")
    axes[1].set_xlabel("Axis 2")
    axes[2].set_ylabel("Axis 0")
    axes[2].set_xlabel("Axis 1")
    
    plt.suptitle('Orthogonal Core View')
    # fig.set_figwidth = 8

    # plt.subplots_adjust(wspace = 120)
    plt.tight_layout()


def display_slice(slice_2d: np.ndarray) -> AxesImage:
    slice_image: plt.AxesImage = plt.imshow(slice_2d)
    return slice_image

def display_slice_bt_std(slice_2d: np.ndarray, title: str = None) -> plt.Axes:
    bt_df = brightness_trace(slice_2d)
    brightness = bt_df.iloc[:,0]
    stddev = bt_df.iloc[:,1]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize = (7,5), width_ratios=[2,3,3])
    ax1.imshow(slice_2d)
    ax1.set_xlabel("width (pixels)")
    ax1.set_ylabel("depth (pixels)")
    ax2.plot(brightness, range(len(bt_df)))
    ax2.set_xlabel("mean brightness (HU)")
    ax3.set_xlabel("brightness standard deviation (HU)")
    ax3.plot(stddev, range(len(bt_df)))
    fig.suptitle(title)
    fig.tight_layout()
    return ax1, ax2, ax3

def visualize_trim(slice_2d: np.ndarray, axis: str, loc: int) -> None:
    """
    overlays trim lines onto a slice to illustrate what is being trimmed
    """
    plt.figure()
    plt.imshow(slice_2d)
    if axis == 'x':
        plt.axvline(loc)
        plt.axvline(len(slice_2d[0]-loc))
    else:
        plt.axhline(loc)
        plt.axhline(len(slice_2d)-loc)
    plt.show()

# be able to visualize graphs and slices separately
# be able to visualize trim lines on core and slices
# visualize orthogonal view of the cores to see what the axes correspond to on the core
# will help you decide how you want to take your slices
# return matplotlib objects that can be modifiable rather than just plot things
