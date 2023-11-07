"""Methods that assist in visualizing the CT scans of rock cores."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from core_ct.core import Core
from .analysis import brightness_trace

def display_core(core: Core) -> (matplotlib.figure.Figure, matplotlib.axes.Axes):
    """
    Create an orthogonal view of the core to aid in understanding the core orientation.

    Displays three cross-sections of the core object in one figure.
    Each slice is taken at the center of the collapsed axis for convenience.

    Arguments:
    ---------
        core: `Core` object to visualize
    
    Returns:
    -------
        `matplotlib.figure.Figure` object containing the subplots
        Array of `matplotlib.axes.Axes` objects, each axis containing display 
        information for each view of the core
    """
    core_dim = core.pixel_array.shape
    fig, axes = plt.subplots(nrows = 1, ncols = 3)
    for i, ax in enumerate(axes):
        slice_loc = core_dim[i]//2
        ax.imshow(core.slice(axis = i, loc = slice_loc))
        ax.set_title("Axis {} collapsed".format(i))
    axes[0].set_ylabel("Axis 1")
    axes[0].set_xlabel("Axis 2")
    axes[1].set_ylabel("Axis 0")
    axes[1].set_xlabel("Axis 2")
    axes[2].set_ylabel("Axis 0")
    axes[2].set_xlabel("Axis 1")
    
    fig.suptitle('Orthogonal Core View')
    fig.tight_layout()
    return fig, axes


def display_slice(slice_2d: np.ndarray) -> matplotlib.image.AxesImage:
    """
    Display an image of a slice of a core using matplotlib's `imshow` function.
    
    Plots a colorbar alongside the slice showing the range of pixel values in the slice.

    Arguments:
    ---------
        slice_2d: 2D numpy array of pixel data for a single slice of a core 

    Returns:
    -------
        `matplotlib.image.AxesImage` object returned by `imshow`
    """
    fig = plt.figure()
    img = plt.imshow(slice_2d)
    cbar = fig.colorbar(img)
    cbar.minorticks_on()
    return img

def display_slice_bt_std(slice_2d: np.ndarray) -> (matplotlib.figure.Figure, 
                                                    matplotlib.pyplot.Axes):
    """
    Display a core slice and corresponding brightness trace and standard deviation.

    Arguments:
    ---------
        slice_2d: 2D numpy array of pixel data for a single slice of a core

    Returns:
    -------
        `matplotlib.figure.Figure` object containing the subplots
        Array of `matplotlib.axes.Axes` objects, each axis containing display 
        information or data for each plot
    """
    bt_df = brightness_trace(slice_2d)
    brightness = bt_df.iloc[:,0]
    stddev = bt_df.iloc[:,1]
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, 
                                        ncols = 3, 
                                        sharey = True, 
                                        width_ratios = [1.5,3,3.5])
    colors = ax1.imshow(slice_2d) # set to a variable to use for colormap
    ax1.set_xlabel("width (pixels)")
    ax1.set_ylabel("depth (pixels)")
    ax2.plot(brightness, range(len(bt_df)))
    ax2.set_xlabel("mean brightness (HU)")
    ax3.set_xlabel("brightness standard deviation (HU)")
    ax3.plot(stddev, range(len(bt_df)))
    cbar = fig.colorbar(colors, ax = ax3)
    cbar.minorticks_on()
    fig.suptitle("Core CT Scan Brightness Trace")
    fig.tight_layout()
    return fig, (ax1, ax2, ax3)

def visualize_trim(slice_2d: np.ndarray, axis: int, loc_start: int, 
                    loc_end: int | None = None) -> matplotlib.image.AxesImage:
    """
    Overlay trim lines onto a slice to illustrate where a trim would occur.

    Arguments:
    ---------
        slice_2d: 2D numpy array of pixel data for a single slice of a core
        axis: integer either 0 or 1 indicating what axis to display the trim on
            0 corresponds to the y axis (row), so a horizontal line will be plotted
            1 corresponds to the x axis (column), so a vertical line will be plotted
        loc_start: integer index specifying where the first line will be plotted 
        loc_end: if given, is an integer specifying where the second line will be 
        plotted as a distance from the end of the axis. Therefore the actual index 
        will be `len(axis)-loc_end`. If not given, loc_end is equal to loc_start 
        so the trim will be symmetric

    Returns:
    -------
        `matplotlib.image.AxesImage` object returned by `imshow`   

    Raises:
    ------
        ValueError if axis is a value other than 0 or 1
    """
    if loc_end is None:
            loc_end = loc_start

    plt.figure()
    match axis:
        case 0:
            plt.axhline(loc_start, color = 'r')
            plt.axhline(len(slice_2d[0])-loc_end, color = 'r')
        case 1:
            plt.axvline(loc_start, color = 'r')
            plt.axvline(len(slice_2d[0])-loc_end, color = 'r')
        case _:
            raise ValueError("axis must be either 0 or 1")
    return plt.imshow(slice_2d)
