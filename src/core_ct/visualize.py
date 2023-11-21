"""Methods that assist in visualizing the CT scans of rock cores."""

import matplotlib
import matplotlib.pyplot as plt
from core_ct.core import Core
from core_ct.slice import Slice
from core_ct.analysis import brightness_trace


def display_core(
    core: Core, mm: bool = False
) -> (matplotlib.figure.Figure, matplotlib.axes.Axes):
    """
    Create an orthogonal view of the core to aid in understanding the core orientation.

    Displays three cross-sections of the core object in one figure.
    Each slice is taken at the center of the collapsed axis for convenience. Option to
    have axes shown in pixels or millimeters.

    Arguments
    ---------
    core : Core
        `Core` object to visualize
        
    mm : bool
        If set to `True`, will show plot axes in mm - default is pixels

    Returns
    -------
    matplotlib.figure.Figure
        Object containing the subplots

    list[matplotlib.axes.Axes]
        Each axis contains display information for each view of the core
    """
    fig, axes = plt.subplots(nrows=1, ncols=3)
    core_dim = core.shape()
    units = "(pixels)"
    for i, ax in enumerate(axes):
        slice_loc = core_dim[i] // 2
        slice = core.slice(axis=i, loc=slice_loc)
        slice_dim = slice.shape()
        # because of row-col conventions, first index of pixel_dimensions is on y
        # and second index of pixel dimensions is on x
        # args to extent are x start, x end, y start, y end
        if mm:  # uses pixel_dimensions as a scalar multiple for pixels
            units = "(mm)"
            ax.imshow(
                slice.data,
                extent=(
                    0,
                    slice_dim[1] * slice.pixel_dimensions[1],
                    slice_dim[0] * slice.pixel_dimensions[0],
                    0,
                ),
            )
        else:
            ax.imshow(slice.data)
        ax.set_title("Axis {} collapsed".format(i))

    axes[0].set_ylabel("Axis 1 {}".format(units))
    axes[0].set_xlabel("Axis 2 {}".format(units))
    axes[1].set_ylabel("Axis 0 {}".format(units))
    axes[1].set_xlabel("Axis 2 {}".format(units))
    axes[2].set_ylabel("Axis 0 {}".format(units))
    axes[2].set_xlabel("Axis 1 {}".format(units))

    fig.suptitle("Orthogonal Core View")
    fig.tight_layout()
    return fig, axes


def display_slice(slice: Slice, mm: bool = False) -> matplotlib.image.AxesImage:
    """
    Display an image of a single slice of a core using matplotlib's `imshow` function.

    Plots a colorbar alongside the slice showing the range of pixel values in the slice.
    Option to have axes shown in pixels or millimeters.

    Arguments
    ---------
    slice : Slice
        `Slice` object to display

    mm : bool
        If set to `True`, will show plot axes in mm - default is pixels

    Returns
    -------
    matplotlib.image.AxesImage
        Object returned by `imshow`
    """
    fig = plt.figure()
    slice_dim = slice.shape()
    units = "(pixels)"
    if mm:
        units = "(mm)"
        img = plt.imshow(
            slice.data,
            extent=(
                0,
                slice_dim[1] * slice.pixel_dimensions[1],
                slice_dim[0] * slice.pixel_dimensions[0],
                0,
            ),
        )
    else:
        img = plt.imshow(slice.data)

    plt.xlabel("Width {}".format(units))
    plt.ylabel("Depth {}".format(units))
    cbar = fig.colorbar(img)
    cbar.minorticks_on()
    return img


def display_slice_bt_std(
    slice: Slice, mm: bool = False
) -> (matplotlib.figure.Figure, matplotlib.pyplot.Axes):
    """
    Display a core slice and corresponding brightness trace and standard deviation.

    Arguments
    ---------
    slice : Slice
        `Slice` object to display

    mm : bool
        If set to `True`, will show plot axes in mm - default is pixels

    Returns
    -------
    matplotlib.figure.Figure
        Object containing the subplots

    list[matplotlib.axes.Axes]
        Each axis contains display information for each view of the core
    """
    bt_df = brightness_trace(slice)
    brightness = bt_df.iloc[:, 0]
    stddev = bt_df.iloc[:, 1]
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, sharey=True, width_ratios=[1.5, 3, 3.5]
    )
    slice_dim = slice.shape()
    pixel_dim = slice.pixel_dimensions
    units = "(pixels)"
    y_data = list(range(slice_dim[0]))  # values for y axis of brightness and std plots
    if mm:
        units = "(mm)"
        y_data = [
            y * pixel_dim[0] for y in range(slice_dim[0])
        ]  # scale according to mm
        img = ax1.imshow(
            slice.data,
            extent=(
                0,
                slice_dim[1] * pixel_dim[1],
                slice_dim[0] * pixel_dim[0],
                0,
            ),
        )
    else:
        img = ax1.imshow(slice.data)

    ax1.set_xlabel("width {}".format(units))
    ax1.set_ylabel("depth {}".format(units))

    # plot brightness graph
    ax2.plot(brightness, y_data)
    ax2.set_xlabel("mean brightness (HU)")
    # plot standard deviation graph
    ax3.plot(stddev, y_data)
    ax3.set_xlabel("standard deviation (HU)")
    # add colorbar legend
    cbar = fig.colorbar(img, ax=ax3)
    cbar.minorticks_on()
    fig.suptitle("Core CT Scan Brightness Trace")
    fig.tight_layout()
    return fig, (ax1, ax2, ax3)


def visualize_trim(
    slice: Slice, axis: int, loc_start: int, loc_end: int | None = None
) -> matplotlib.image.AxesImage:
    """
    Overlay trim lines onto a slice to illustrate where a trim would occur.

    Arguments
    ---------
    slice : Slice
        `Slice` object to visualize

    axis : int
        Integer either 0 or 1 indicating what axis to display the trim on

            0: corresponds to the y axis (row), so a horizontal line will be plotted

            1: corresponds to the x axis (column), so a vertical line will be plotted

    loc_start : int
        Integer index specifying where the first line will be plotted

    loc_end : int
        If given, is an integer specifying where the second line will be plotted as a
        distance from the end of the axis. Therefore the actual index will be
        `len(axis)-loc_end`. If not given, loc_end is equal to loc_start so the trim
        will be symmetric

    Returns
    -------
    matplotlib.image.AxesImage
        Object returned by `imshow`

    Raises
    ------
    ValueError
        If axis is a value other than 0 or 1

    IndexError
        If amount trimmed from end causes the ending index to be to the
        left of the starting index

    IndexError
        If start_loc or end_loc is out of bounds of the axis length
    """
    slice_dim = slice.shape()
    if axis != 0 and axis != 1:
        raise ValueError("axis must be an integer either 0 or 1")
    if loc_end is None:
        loc_end = loc_start
    if slice_dim[axis] - loc_end < loc_start:
        raise IndexError("starting index exceeds ending index")

    plt.figure()
    if axis == 0:
        plt.axhline(loc_start, color="r")
        plt.axhline(slice_dim[0] - loc_end, color="r")
    else:  # axis == 1
        plt.axvline(loc_start, color="r")
        plt.axvline(slice_dim[1] - loc_end, color="r")
    return plt.imshow(slice.data)
