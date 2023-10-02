from core_ct import Core
import numpy as np


def test_core():
    # Define the pixel array and pixel_dimensions
    pixel_array = np.zeros([2, 4, 8])
    pixel_dimensions = [2.0, 4.0, 8.0]

    # Create the core object
    core = Core(pixel_array=pixel_array, pixel_dimensions=pixel_dimensions)

    # Assert the core was created correctly
    np.testing.assert_equal(core.pixel_array, pixel_array)
    np.testing.assert_equal(core.pixel_dimensions, pixel_dimensions)

def test_slice():
    # Define the core
    core = Core(pixel_array=np.zeros([2, 4, 8]), pixel_dimensions=[2.0, 4.0, 8.0])

    # Take slice from each axis out of the core
    slice_0 = core.slice(axis=0, loc=0)
    slice_1 = core.slice(axis=1, loc=0)
    slice_2 = core.slice(axis=2, loc=0)

    # Check that the shape of each slice is correct
    assert slice_0.shape == (4, 8)
    assert slice_1.shape == (2, 8)
    assert slice_2.shape == (2, 4)
    