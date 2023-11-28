"""A class that abstracts the CT scan of a rock core."""

from __future__ import annotations
import numpy as np
from typing import Callable
from core_ct.slice import Slice
from math import pow, sqrt

# Methods
# -------
#     slice(self, axis, loc) -- get a 2D slice of the core


class Core:
    """
    Abstracts properties of a core CT-scan and methods for manipulating it.

    Attributes
    ----------
    data : np.ndarray
        3D numpy array of pixel data that make up the core

    pixel_dimensions : tuple[int, int , int]
        tuple containing the dimensions of each pixel/voxel
    """

    def __init__(
        self,
        data: np.ndarray | list[float],
        pixel_dimensions: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """
        Construct necessary attributes of a Core.

        Arguments
        ---------
        data : `np.ndarray`
            3D numpy array of pixel data that make up the core

        pixel_dimensions : `tuple[int, int, int]`
            tuple containing the dimensions of each pixel/voxel
        """
        self.pixel_dimensions: tuple[float, float, float] = pixel_dimensions

        # data must be in a numpy array for slicing methods to work
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data

    def slice(self, axis: int, loc: int) -> Slice:
        """
        Get a 2D `Slice` of the core at a specific location along an axis.

        Arguments
        ---------
        axis : int
            Integer either 0, 1, 2 specifying which dimension to collapse

                0: corresponds to x-axis

                1: corresponds to y-axis

                2: corresponds to z-axis

        loc : int
            Integer value along the axis specifying the location of the slice

        Returns
        -------
        Slice
            `Slice` object containing pixel data and dimensions

        Raises
        ------
        ValueError
            If axis is a value other than 0, 1, or 2
        """
        match axis:
            case 0:
                return Slice(
                    data=self.data[loc],
                    pixel_dimensions=(
                        self.pixel_dimensions[1],
                        self.pixel_dimensions[2],
                    ),
                )  # 0th and 1st
            case 1:
                return Slice(
                    data=self.data[:, loc],
                    pixel_dimensions=(
                        self.pixel_dimensions[0],
                        self.pixel_dimensions[2],
                    ),
                )
            case 2:
                return Slice(
                    data=self.data[:, :, loc],
                    pixel_dimensions=(
                        self.pixel_dimensions[0],
                        self.pixel_dimensions[1],
                    ),
                )
            case _:
                raise ValueError("axis must be a value between 0 and 2 (inclusive)")

    def trim(self, axis: int, loc_start: int, loc_end: int | None = None) -> Core:
        """
        Reduces the dimensions of the core along a specified axis.

        Get a three-dimensional slice of the core scan by trimming off a specified
        amount on the requested axis. This function is symmetrical by default.

        Arguments
        ---------
        axis : int
            Integer either 0, 1, 2 specifying which dimension to collapse:

                0: corresponds to x-axis

                1: corresponds to y-axis

                2: corresponds to z-axis

        loc_start : int
            Specifies the amount to trim off the beginning

        loc_end : int
            Specifies the amount to trim off the end

        Returns
        -------
        Core
            A new trimmed core object

        Raises
        ------
        ValueError
            If axis is a value other than 0, 1, or 2
        """
        if loc_end is None:
            loc_end = loc_start

        match axis:
            case 0:
                new_data = self.data[loc_start : len(self.data) - loc_end]
            case 1:
                new_data = self.data[:, loc_start : len(self.data[0]) - loc_end]
            case 2:
                new_data = self.data[:, :, loc_start : len(self.data[0, 0]) - loc_end]
            case _:
                raise ValueError("axis must be a value between 0 and 2 (inclusive)")

        return Core(new_data, self.pixel_dimensions)

    def trim_by_percent(
        self, axis: int, percent_start: float, percent_end: float | None = None
    ) -> Core:
        """
        Reduces the dimensions of the core along a specified axis.

        Get a three-dimensional slice of the core scan by trimming off a percent
        on the requested axis. This function is symmetrical by default, trimming 
        `percent_start` from each side.

        Arguments
        ---------
        axis : int
            integer either 0, 1, 2 specifying which dimension to collapse:

                0: corresponds to x-axis

                1: corresponds to y-axis

                2: corresponds to z-axis

        percent_start : float
            Percent to trim off the left side

        percent_end : float
            Percent to trim off the right side

        Returns
        -------
        Core
            A new trimmed core object

        Raises
        ------
        ValueError
            If axis is a value other than 0, 1, or 2
        """
        if percent_end is None:
            percent_end = percent_start

        if percent_start + percent_end > 1.0:
            raise ValueError("Percents must be a less than 1.0")

        if axis in [0, 1, 2]:
            loc_start = int(self.data.shape[axis] * percent_start)
            loc_end = int(self.data.shape[axis] * percent_end)
        else:
            raise ValueError("axis must be a value between 0 and 2 (inclusive)")

        return self.trim(axis, loc_start, loc_end)

    def trim_radial(
        self,
        axis: int,
        radius: float,
        x_center: int | None = None,
        y_center: int | None = None,
        z_center: int | None = None,
    ) -> Core:
        """
        Trims the Core radially given an axis and a center.

        Replaces all data outside of the user specified area with NaN. Also reduces the
        dimensions of `data` as much as possible.

        The user specifies a cylindrical shape by an `axis` and a center. For example,
        if `axis` is set to `2` (z-axis) the user should specify the center via
        `x_center` and `y_center`. After trimming, every z-slice will only contain
        data within a circle of the given `radius` centered at (`x_center`, `y_center`).

        This function is inclusive on `radius`, i.e. the returned Core maintains data 
        where distance from center is equal to `radius`.

        The user does not need to specify the `center` argument for their chosen `axis`.
        All `center` arguments default to the middle of their respective axes if not 
        provided.

        Parameters
        ----------
        axis : int
            axis to radially trim about

                0 corresponds to x-axis

                1 corresponds to y-axis

                2 corresponds to z-axis

        radius : float
            radius from given center to trim values outside of

        x_center : int
            index to center the cylinder on along the x-axis

        y_center : int
            index to center the cylinder on along the y-axis

        z_center : int
            index to center the cylinder on along the z-axis

        Returns
        -------
        Core
            A new trimmed core object

        Raises
        ------
        ValueError
            If axis is a value other than 0, 1, or 2
        """
        # figure out which axis we are testing the radius against
        dist_axis_1: int
        dist_axis_2: int
        match axis:
            case 0:
                dist_axis_1 = 1
                dist_axis_2 = 2
            case 1:
                dist_axis_1 = 0
                dist_axis_2 = 2
            case 2:
                dist_axis_1 = 0
                dist_axis_2 = 1
            case _:
                raise ValueError("axis must be a value between 0 and 2 (inclusive)")

        # clean up center inputs
        if x_center is None:
            x_center = int(self.data.shape[0] / 2)
        if y_center is None:
            y_center = int(self.data.shape[1] / 2)
        if z_center is None:
            z_center = int(self.data.shape[2] / 2)

        center: tuple[int, int, int] = (x_center, y_center, z_center)

        starts: list[int] = [0] * 3
        ends: list[int] = [0] * 3
        for ax in range(0, 3):
            if ax == axis:
                starts[ax] = 0
                ends[ax] = self.data.shape[ax]
                continue

            pixel_radius = int(radius / self.pixel_dimensions[ax])
            starts[ax] = int(max(center[ax] - pixel_radius, 0))
            ends[ax] = int(min(center[ax] + pixel_radius, self.data.shape[ax])) + 1

        # must create a copy instead of a view because we are destructively modifying
        # data during the filter step
        data: np.ndarray = self.data[
            starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]
        ].copy()

        # should calculate radius in terms of our new reduced matrix, move center
        # accordingly
        center = (center[0] - starts[0], center[1] - starts[1], center[2] - starts[2])

        # filter out all data outside the radius
        for idx_1 in range(data.shape[dist_axis_1]):
            for idx_2 in range(data.shape[dist_axis_2]):
                # find distance from center to this point using pixel_dimensions
                dist_1: float = (center[dist_axis_1] - idx_1) \
                    * self.pixel_dimensions[dist_axis_1]
                dist_2: float = (center[dist_axis_2] - idx_2) \
                    * self.pixel_dimensions[dist_axis_2]
                dist: float = sqrt(pow(dist_1, 2) + pow(dist_2, 2))

                # only want to erase data outside radius
                if dist <= radius:
                    continue

                # find indices of data we want to erase
                indices = [None] * 3
                indices[dist_axis_1] = idx_1
                indices[dist_axis_2] = idx_2
                indices[axis] = slice(None)
                # convert indices to tuple to be useable in a numpy subscript operation
                indices = tuple(indices)

                # indices will be unwrapped when used in a numpy subscript operation
                # for example, if we assume:
                #   dist_axis_1 = 0
                #   dist_axis_2 = 1
                #   axis = 2
                #   idx_1 = 4
                #   idx_2 = 8
                # then:
                #   indices = (4, 8, slice(None))
                # which when used on a numpy array like this:
                #   a = data[indices]
                # is equivalent to this:
                #   a = data[4, 8, :]

                # create view containing data we want to erase
                view: np.ndarray = data[indices]
                # erase data contained in view
                view.fill(np.nan)

        return Core(data=data, pixel_dimensions=self.pixel_dimensions)

    def swapaxes(self, axis1: int, axis2: int) -> Core:
        """
        Create a new `Core` object with swapped axes and updated pixel dimensions.

        Arguments
        ---------
        axis1 : int
            Integer either 0, 1, 2 specifying the first axis:

                0: corresponds to x-axis

                1: corresponds to y-axis

                2: corresponds to z-axis

        axis2 : int
            Integer either 0, 1, 2 specifying the second axis:

                0: corresponds to x-axis

                1: corresponds to y-axis

                2: corresponds to z-axis

        Returns
        -------
        Core
            New Core object containing swapped data and updated pixel dimensions

        Raises
        ------
        ValueError
            If axes are values other than 0, 1, or 2
        """
        # make sure axis inputs are valid
        if axis1 < 0 or axis1 > 2:
            raise ValueError("axis1 must be a value between 0 and 2 (inclusive)")
        if axis2 < 0 or axis2 > 2:
            raise ValueError("axis2 must be a value between 0 and 2 (inclusive)")

        # swap axes in data array
        data = np.swapaxes(self.data, axis1, axis2)

        # swap values in pixel dimensions
        pixel_dimensions: list[float] = list(self.pixel_dimensions)
        pixel_dimensions[axis1] = self.pixel_dimensions[axis2]
        pixel_dimensions[axis2] = self.pixel_dimensions[axis1]

        # return new Core containing transformed data
        return Core(data=data, pixel_dimensions=tuple(pixel_dimensions))

    def flip(self, axis: int) -> Core:
        """
        Create a new `Core` object with data reversed along the given axis.

        Arguments
        ---------
        axis : int
            Integer either 0, 1, 2 specifying which axis to reverse

                0: corresponds to x-axis

                1: corresponds to y-axis

                2: corresponds to z-axis

        Returns
        -------
        Core
            New Core object containing flipped data

        Raises
        ------
        ValueError
            If axis is a value other than 0, 1, or 2
        """
        # make sure axis inputs are valid
        if axis < 0 or axis > 2:
            raise ValueError("axis must be a value between 0 and 2 (inclusive)")

        # swap axes in data array
        data = np.flip(self.data, axis)

        # return new Core containing transformed data
        return Core(data=data, pixel_dimensions=self.pixel_dimensions)

    def rotate(self, axis: int, k: int = 1, clockwise: bool = False) -> Core:
        """
        Create a new `Core` object with data rotated 90 degrees about `axis` `k` times.

        Rotates counter-clockwise by default, set `clockwise` to `True` to rotate
        clockwise instead.

        Arguments
        ---------
        axis : int
            Integer either 0, 1, 2 specifying which axis to rotate about

                0: corresponds to x-axis

                1: corresponds to y-axis

                2: corresponds to z-axis

        k : int
            Number of times to rotate pixel_array 90 degrees

        clockwise : bool
            whether or not to rotate clockwise instead of counter-clockwise

        Returns
        -------
        Core
            New Core object containing rotated data and pixel dimensions

        Raises
        ------
        ValueError
            If axis is a value other than 0, 1, or 2
        """
        # make sure axis inputs are valid
        if axis < 0 or axis > 2:
            raise ValueError("axis must be a value between 0 and 2 (inclusive)")

        # handle clockwise/counter-clockwise conversion
        if clockwise:
            k = -k

        # figure out which axis to use in call to numpy.rot90()
        axis1: int
        axis2: int

        match axis:
            case 0:
                axis1 = 1
                axis2 = 2
            case 1:
                axis1 = 0
                axis2 = 2
            case 2:
                axis1 = 0
                axis2 = 1

        data = np.rot90(self.data, k=k, axes=(axis1, axis2))

        # correcting pixel_dimensions below the rot90 call so pixel_dimensions won't
        # be messed up if rot90 fails

        # figure out how to modify pixel_dimensions
        # if k is even, the array is being rotated by a factor of 180 degrees so we
        # don't need to worry about switching dimensions
        pixel_dimensions: list[float] = list(self.pixel_dimensions)
        if k % 2 != 0:
            # swap dimensions of correct axes
            pixel_dimensions[axis1] = self.pixel_dimensions[axis2]
            pixel_dimensions[axis2] = self.pixel_dimensions[axis1]

        # return new Core with transformed data
        return Core(data=data, pixel_dimensions=tuple(pixel_dimensions))

    def chunk(self, x1=0, y1=0, z1=0, x2=None, y2=None, z2=None) -> Core:
        """
        Get a three-dimensional section of the core scan.

        Arguments
        ---------
            x1 : int
                The starting x position for the chunk to be taken

            y1 : int
                The starting y position for the chunk to be taken

            z1 : int
                The starting z position for the chunk to be taken

            x2 : int
                The ending x position for the chunk to be taken

            y2 : int
                The ending y position for the chunk to be taken

            z2 : int
                The ending z position for the chunk to be taken

        Returns
        -------
        Core
            New core object containing the specified chunk of the old core
        """
        if x2 is None:
            x2 = len(self.data)
        if y2 is None:
            y2 = len(self.data[0])
        if z2 is None:
            z2 = len(self.data[0, 0])

        # Make sure that the first value smaller
        if x2 < x1:
            temp = x1
            x1 = x2
            x2 = temp
        if y2 < y1:
            temp = y1
            y1 = y2
            y2 = temp
        if z2 < z1:
            temp = z1
            z1 = z2
            z2 = temp

        new_core = Core(self.data[x1:x2, y1:y2, z1:z2], self.pixel_dimensions)
        return new_core

    def shape(self) -> tuple[int, int, int]:
        """
        Get the dimensions of the pixel array of the core scan.

        Returns
        -------
        tuple[int, int, int]
            The pixel dimensions of the core scan.
        """
        return self.data.shape

    def dimensions(self) -> tuple[float, float, float]:
        """
        Get the dimensions of the scan in mm.

        Returns
        -------
        tuple[int, int, int]
            A three-element tuple containing the dimensions of the scan in mm.
        """
        return tuple(
            size * dimension
            for size, dimension in zip(self.data.shape, self.pixel_dimensions)
        )

    def volume(self) -> float:
        """
        Approximates the core volume in mm; ignores any NaN values.

        Returns
        -------
        float
            The approximate volume of the core in cubic mm ignoring NaN values.
        """
        # Calculate the volume of a voxel
        voxel_volume = (
            self.pixel_dimensions[0]
            * self.pixel_dimensions[1]
            * self.pixel_dimensions[2]
        )

        # Count the number of voxels within the density range
        valid_voxels = (~np.isnan(self.data)).sum()

        return valid_voxels * voxel_volume

    def filter(self, brightness_filter: Callable[[float], bool]) -> Core:
        """
        Create new Core containing data filtered according to the provided function.

        The filter function (lambda) is run on every brightness value in `data`. If the 
        lambda returns `True`, the new Core will contain that brightness value. If the 
        lambda returns `False`, the new Core will not contain that datapoint, 
        substituting it with `numpy.nan` (`np.nan`).

        This function can take a long time to run depending on how much data the Core
        contains. Consider using `Core.trim()`, `Core.trim_radial()`, or `Core.chunk()`
        to reduce the amount of data you are filtering.

        Arguments
        ---------
        brightness_filter : Callable[[float], bool]
            Lambda function that defines what will be filtered out. Function must either
            return `False` if the value should not be included or `True` if the value 
            should be included.

        Returns
        -------
        Core
            New core object with only matching brightness values left,
            everything else is set to `numpy.nan` (`np.nan`).
        """
        filter_lambda = np.vectorize(lambda x: x if brightness_filter(x) else np.nan)
        filtered = filter_lambda(self.data)

        return Core(data=filtered, pixel_dimensions=self.pixel_dimensions)

    def join(self, core: Core, axis: int = 0) -> Core:
        """
        Join a core to the current core on a specified axis.

        Arguments
        ---------
        core : Core
            The `Core` object to join with the current core

        axis : int
            Integer either 0, 1, 2 specifying which axis to join on

                0: corresponds to x-axis

                1: corresponds to y-axis

                2: corresponds to z-axis

        Returns
        -------
        Core
            New core object made up of the two joined arrays

        Raises
        ------
        ValueError
            If axis is a value other than 0, 1, or 2

        ValueError
            If the `pixel_dimensions` of the cores don't match

        ValueError
            If the shapes of the cores along an axis don't match
        """
        # Check that the axis values are valid
        if axis < 0 or axis > 2:
            raise ValueError("axis must be a value between 0 and 2 (inclusive)")

        # Check that the pixel dimensions match between the two cores
        if core.pixel_dimensions != self.pixel_dimensions:
            raise ValueError(
                "the core's pixel dimensions must match, {source} != {target}".format(
                    source=core.pixel_dimensions, target=self.pixel_dimensions
                )
            )

        # Join the two data arrays together
        joined_data = np.append(self.data, core.data, axis=axis)

        return Core(joined_data, self.pixel_dimensions)
