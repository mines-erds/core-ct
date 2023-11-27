"""A class that abstracts a 2D slice of a `Core`."""
from __future__ import annotations
import numpy as np


class Slice:
    """
    Abstracts properties of a core slice and contains methods for manipulating it.

    Attributes
    ----------
    data : np.ndarray
        2D numpy array of pixel data that make up the slice

    pixel_dimensions : tuple[float, float]
        Tuple containing the dimensions of each pixel as (width, height) in mm
    """

    def __init__(self, data: np.ndarray, pixel_dimensions: tuple[float, float]):
        """
        Construct necessary attributes of a core slice.

        Arguments
        ---------
            data : np.ndarray
                2D numpy array of pixel data that make up the slice

        pixel_dimensions : tuple[float, float]
            Tuple containing the dimensions of each pixel as (width, height) in mm
        """
        self.data: np.ndarray = data
        self.pixel_dimensions: tuple[float, float] = pixel_dimensions

    def trim(self, axis: int, loc_start: int, loc_end: int | None = None) -> Slice:
        """
        Create new slice by trimming off a specified amount on the requested axis.

        Remove unwanted space around a core that must be removed before analysis.
        A new `Slice` object will be created by taking a slice of the data array
        between indices `start_loc` and `len(axis) - end_loc`.
        By default this function is symmetric so the same amount will be taken from
        either ends of the specified axis. The new slice has the same pixel dimensions.

        Arguments
        ---------
            axis : int
                Integer either 0 or 1 specifying which axis to trim from

                0: corresponds to the y axis (row) so it trims horizontally

                1: corresponds to the x axis (column) so it trims vertically

            loc_start : int
                Integer specifying the amount to trim off from the start of the axis

            loc_end : int
                If given, is an integer specifying where the second trim will occur as a
                distance from the end of the axis. Therefore the actual index will be
                `len(axis)-loc_end`. If not given, loc_end is equal to loc_start so the
                trim will be symmetric

        Returns
        -------
        Slice
            A new trimmed `Slice` object

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
        if axis != 0 and axis != 1:
            raise ValueError("axis must be an integer either 0 or 1")
        if loc_end is None:
            loc_end = loc_start
        if self.data.shape[axis] - loc_end < loc_start:
            raise IndexError("starting index exceeds ending index")

        if axis == 0:
            new_data_array = self.data[loc_start : self.data.shape[0] - loc_end, :]
        else:  # axis == 1
            new_data_array = self.data[:, loc_start : self.data.shape[1] - loc_end]

        return Slice(new_data_array, self.pixel_dimensions)

    def shape(self) -> tuple[int, int]:
        """
        Get the dimensions of the `data` array of the core slice.

        Returns
        -------
        tuple[int, int]
            The shape of the data array of the core slice.
        """
        return self.data.shape
