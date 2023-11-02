"""Methods that assist in visualizing the CT scans of rock cores."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from core_ct.core import Core
from .analysis import brightness_trace

def display_core(core: Core) -> (matplotlib.figure.Figure, matplotlib.axes.Axes):
    """
    Create an orthogonal view of the core to aid in understanding the core orientation
    Displays three cross-sections of the core object in one figure, each taken at the center of the collapsed axis.

    Arguments: 
    ----------
        core: `Core` object to visualize
    
    Returns:
    --------
        `matplotlib.figure.Figure` object containing the subplots
        Array of `matplotlib.axes.Axes` objects, each axis containing display information for each view
    """
    core_dim = core.pixel_array.shape
    fig, axes = plt.subplots(nrows = 1, ncols = 3)
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
    
    fig.suptitle('Orthogonal Core View')
    fig.tight_layout()
    plt.show()
    return fig, axes


def display_slice(slice_2d: np.ndarray) -> matplotlib.image.AxesImage:
    """
    Display an image of a slice of a core using matplotlib's `imshow` function

    Arguments:
    ----------
    slice_2d: 2D numpy array of pixel data for a single slice of a core 

    Returns:
    --------
    `matplotlib.image.AxesImage` object returned by `imshow`
    """
    slice_image: plt.AxesImage = plt.imshow(slice_2d)
    return slice_image

def display_slice_bt_std(slice_2d: np.ndarray, title: str = None) -> (matplotlib.figure.Figure, matplotlib.pyplot.Axes):
    """
    Display a core slice and corresponding brightness trace and standard deviation graphs

    Arguments:
    ----------
    slice_2d: 2D numpy array of pixel data for a single slice of a core 

    Returns:
    --------
    `matplotlib.figure.Figure` object containing the subplots
    Array of `matplotlib.axes.Axes` objects, each axis containing display information or data for each plot
    """
    bt_df = brightness_trace(slice_2d)
    brightness = bt_df.iloc[:,0]
    stddev = bt_df.iloc[:,1]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, sharey = True, figsize = (7,5), width_ratios = [1.5,3,3.5])
    colors = ax1.imshow(slice_2d) # set to a variable to use for colormap
    ax1.set_xlabel("width (pixels)")
    ax1.set_ylabel("depth (pixels)")
    ax2.plot(brightness, range(len(bt_df)))
    ax2.set_xlabel("mean brightness (HU)")
    ax3.set_xlabel("brightness standard deviation (HU)")
    ax3.plot(stddev, range(len(bt_df)))
    cbar = fig.colorbar(colors, ax = ax3)
    cbar.minorticks_on()
    fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax1, ax2, ax3)

def visualize_trim(slice_2d: np.ndarray, axis: str, loc: int) -> None:
    """
    overlays trim lines onto a slice to illustrate what is being trimmed
    trimming along 0 should show collapse of 1 and 2
    trimming along 1 should show collapse of 0 and 2
    trimming along 2 should show collapse of 0 and 1
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
