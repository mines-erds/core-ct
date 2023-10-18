"""A class that abstracts the CT scan of a rock core."""

from __future__ import annotations
import numpy as np


class Core:
    """
    Abstracts properties of a core CT-scan and methods for manipulating it.

    Attributes
    ----------
        pixel_array -- 3D numpy array of pixel data that make up the core
        pixel_dimensions - list containing the dimensions of each pixel/voxel

    Methods
    -------
        slice(self, axis, loc) -- get a 2D slice of the core
    """

    def __init__(
        self,
        pixel_array: np.ndarray | list[float],
        pixel_dimensions: list[float] = [1.0, 1.0, 1.0],
    ):
        """
        Construct necessary attributes of a Core.

        Arguments:
        ---------
            pixel_array: 3D numpy array of pixel data that make up the core
            pixel_dimensions: list containing the dimensions of each pixel/voxel
        """
        self.pixel_dimensions: list[float] = pixel_dimensions

        # data must be in a numpy array for slicing methods to work
        if not isinstance(pixel_array, np.ndarray):
            self.pixel_array = np.array(pixel_array)
        else:
            self.pixel_array = pixel_array

    def slice(self, axis: int, loc: int) -> np.ndarray:
        """
        Get a two-dimensional slice of the core at a specific location along an axis.

        Arguments:
        ---------
            axis: integer either 0,1,2 specifying which dimension to collapse:
                    0 corresponds to x-axis
                    1 corresponds to y-axis
                    2 corresponds to z-axis
            loc: integer value along the axis specifying the location of the slice

        Returns:
        -------
            2D numpy array representing a single slice of the core

        Raises:
        ------
            Exception if axis is a value other than 0, 1, or 2
        """
        match axis:
            case 0:
                return self.pixel_array[loc]
            case 1:
                return self.pixel_array[:, loc]
            case 2:
                return self.pixel_array[:, :, loc]
            case _:
                raise Exception("axis must be a value between 0 and 2 (inclusive)")

    def trim(self, axis: int, loc_start: int, loc_end: int | None = None) -> None:
        """
        Reduces the dimensions of the core along a specified axis.

        Get a three-dimensional slice of the core scan by trimming off a specified
        amount on the requested axis. This function is symmetrical by default.

        Arguments:
        ---------
            axis: integer either 0,1,2 specifying which dimension to collapse:
                    0 corresponds to x-axis
                    1 corresponds to y-axis
                    2 corresponds to z-axis
            loc_start: specifies the amount to trim off the beginning.
            loc_end: specifies the amount to trim off the end.

        Raises:
        ------
            Exception if axis is a value other than 0, 1, or 2
        """
        if loc_end is None:
            loc_end = loc_start

        match axis:
            case 0:
                self.pixel_array = self.pixel_array[
                    loc_start : len(self.pixel_array) - loc_end
                ]
            case 1:
                self.pixel_array = self.pixel_array[
                    :, loc_start : len(self.pixel_array[0]) - loc_end
                ]
            case 2:
                self.pixel_array = self.pixel_array[
                    :, :, loc_start : len(self.pixel_array[0, 0]) - loc_end
                ]
            case _:
                raise Exception("axis must be a value between 0 and 2 (inclusive)")

    def chunk(self, x1=0, y1=0, z1=0, x2=None, y2=None, z2=None) -> Core:
        """
        Get a three-dimensional section of the core scan.

        Arguments:
        ---------
            x1: the starting x position for the chunk to be taken
            y1: the starting y position for the chunk to be taken
            z1: the starting z position for the chunk to be taken
            x2: the ending x position for the chunk to be taken
            y2: the ending y position for the chunk to be taken
            z2: the ending z position for the chunk to be taken

        Returns:
        -------
            New core object containing the specified chunk of the old core
        """
        if x2 is None:
            x2 = len(self.pixel_array)
        if y2 is None:
            y2 = len(self.pixel_array[0])
        if z2 is None:
            z2 = len(self.pixel_array[0, 0])

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

        new_core = Core(self.pixel_array[x1:x2, y1:y2, z1:z2], self.pixel_dimensions)
        return new_core
