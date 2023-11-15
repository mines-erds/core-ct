"""Tests the `Core` class in the `core` module."""

import sys
import os

import numpy as np
# import copy
# import pytest

# Add the path to local library folder
local_path = os.path.abspath("src/core_ct")
sys.path.insert(0, os.path.abspath(local_path))

from slice import Slice # noqa

def test_slice():
    """Tests that a `Slice` object can be created successfully."""
    # Define a pixel array and pixel_dimensions
    pixel_array = np.zeros([4, 8])
    pixel_dimensions = [2.0, 4.0]
    
    # Create the slice object
    slice = Slice(data = pixel_array, pixel_dimensions = pixel_dimensions)

    # Assert the slice was created correctly
    np.testing.assert_equal(slice.data, pixel_array)
    np.testing.assert_equal(slice.pixel_dimensions, pixel_dimensions)
