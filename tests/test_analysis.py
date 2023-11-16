"""Tests methods in the `analysis` module."""

from core_ct import analysis
import numpy as np
from core_ct.slice import Slice


def test_brightness_trace():
    """Tests that the `brightness_trace` method correctly computes the mean and std."""
    # Define the slice parameters
    data = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ]
    )
    pixel_dimensions = [2.0, 2.0]
    slice = Slice(data = data, pixel_dimensions = pixel_dimensions)
    # Calculate a brightness trace of the slice
    bt = analysis.brightness_trace(slice)

    # Check that the mean of each row was taken correctly
    assert bt["mean"][0] == np.mean(slice.data[0])
    assert bt["mean"][1] == np.mean(slice.data[1])
    assert bt["mean"][2] == np.mean(slice.data[2])

    # Check that the STD of each row was taken correctly
    assert bt["stddev"][0] == np.std(slice.data[0])
    assert bt["stddev"][1] == np.std(slice.data[1])
    assert bt["stddev"][2] == np.std(slice.data[2])
