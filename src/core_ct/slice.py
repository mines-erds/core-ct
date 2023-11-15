"""A class that abstracts a 2D slice of a `Core`."""
from __future__ import annotations
import numpy as np

class Slice:
    """
    Abstracts properties of a core slice and contains methods for manipulating it.

    Attributes
    ----------
        data -- 2D numpy array of pixel data that make up the slice
        pixel_dimensions -- tuple of (float, float) containing the dimensions of
                            each pixel as (width, height) in mm

    Methods
    -------
        trim(self, axis, loc_start, loc_end = None) -- create new `Slice` with extra
        space along an axis removed

        shape(self) -- get shape of the slice's `data` array
    """

    def __init__(self, data: np.ndarray, pixel_dimensions: tuple[float, float]): 
        """
        Construct necessary attributes of a core slice.

        Arguments:
        ---------
            data: 2D numpy array of pixel data that make up the slice
            pixel_dimensions: tuple of (float, float) containing the dimensions of
                                each pixel as (width, height) in mm
        """
        self.data: np.ndarray = data
        self.pixel_dimensions: tuple[float, float] = pixel_dimensions

    def trim(self, axis: int, loc_start: int, loc_end: int | None = None) -> Slice:  # noqa: D417
        """
        Create new slice by trimming off a specified amount on the requested axis.

        Remove unwanted space around a core that must be removed before analysis.
        A new `Slice` obeject will be created by taking a slice of the data array 
        between indices `start_loc` and `len(axis) - end_loc`. 
        By default this function is symmetric so the same amount will be taken from 
        either ends of the specified axis. The new slice has the same pixel dimensions.

        Arguments:
        ---------
            axis -- integer either 0 or 1 specifying which axis to trim from
                0 - will perform a "horizontal" slice 
                1 - will perform a "vertical" slice
            loc_start -- integer specifying the amount to trim off from the start of the
            axis
            loc_end -- integer specifying the amount to trim off from the end of the 
            axis. Thus the actual index of the trim location is `len(axis)-loc_end`.

        Returns:
        -------
        A new trimmed `Slice` object

        Raises:
        ------
            ValueError if axis is a value other than 0 or 1
            IndexError if amount trimmed from end causes the ending index to be to the
            left of the starting index
        """
        if axis != 0 or axis != 1:
            raise ValueError("axis must be an integer either 0 or 1")
        if self.data.shape[axis] - loc_end < loc_start:
            raise IndexError("starting index exceeds ending index")
        
        if loc_end is None:
            loc_end = loc_start
        if axis == 0:
            new_data_array = self.data[loc_start : self.data.shape[0] - loc_end, :]
        else: # axis == 1
            new_data_array = self.data[:, loc_start : self.data.shape[1] - loc_end]
        
        return Slice(new_data_array, self.pixel_dimensions)

    def shape(self) -> tuple[int, int]:
        """
        Get the dimensions of the `data` array of the core slice.

        Arguments:
        ---------
            none

        Returns:
        -------
            The pixel dimensions of the core slice.
        """
        return self.data.shape