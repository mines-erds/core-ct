"""A class that abstracts the CT scan of a rock core."""

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
        self, pixel_array: np.ndarray, pixel_dimensions: list[float] = [1.0, 1.0, 1.0]
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

    def slice(self, axis, loc):
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
