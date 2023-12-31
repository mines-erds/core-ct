"""Tests the `Slice` class in the `slice` module."""

from core_ct.slice import Slice
import numpy as np


def test_slice():
    """Tests that a `Slice` object can be created successfully."""
    # Define a data array and pixel_dimensions
    data = np.zeros([4, 8])
    pixel_dimensions = [2.0, 4.0]

    # Create the slice object
    slice = Slice(data=data, pixel_dimensions=pixel_dimensions)

    # Assert the slice was created correctly
    np.testing.assert_equal(slice.data, data)
    np.testing.assert_equal(slice.pixel_dimensions, pixel_dimensions)


def test_trim():
    """Tests that a new `Slice` object is created correctly by trimming a slice."""
    # create fake pixel data
    data = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ]
    )
    pixel_dimensions = [2.0, 4.0]
    slice = Slice(data=data, pixel_dimensions=pixel_dimensions)
    # take various slices on each axis, and with and without loc_end
    slice_0 = slice.trim(axis=0, loc_start=2)
    slice_1 = slice.trim(axis=0, loc_start=2, loc_end=3)
    slice_2 = slice.trim(axis=1, loc_start=1)
    slice_3 = slice.trim(axis=1, loc_start=1, loc_end=2)
    # assert shapes are to be expected when trimming along each axis
    assert slice_0.data.shape == (2, 4)
    assert slice_1.data.shape == (1, 4)
    assert slice_2.data.shape == (6, 2)
    assert slice_3.data.shape == (6, 1)
    # assert values are as expected
    assert np.array_equal(slice_0.data, np.array([[9, 10, 11, 12], [13, 14, 15, 16]]))
    assert np.array_equal(slice_1.data, np.array([[9, 10, 11, 12]]))
    assert np.array_equal(
        slice_2.data, np.array([[2, 3], [6, 7], [10, 11], [14, 15], [18, 19], [22, 23]])
    )
    assert np.array_equal(slice_3.data, np.array([[2], [6], [10], [14], [18], [22]]))

    # assert pixel_dimensions are as expected
    assert slice_0.pixel_dimensions == slice.pixel_dimensions
    assert slice_3.pixel_dimensions == slice.pixel_dimensions

    # check that trim raises the proper exceptions
    try:
        slice.trim(axis=2, loc_start=2)  # bad axis
        # if this line is reached, trim failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior)
        assert True
    try:
        slice.trim(axis=0, loc_start=9)  # bad index
        # if this line is reached, trim failed to raise an exception
        assert False
    except IndexError:
        # IndexError was raised (expected behavior)
        assert True
    try:
        slice.trim(axis=1, loc_start=9)  # bad index
        # if this line is reached, trim failed to raise an exception
        assert False
    except IndexError:
        # IndexError was raised (expected behavior)
        assert True

    try:
        slice.trim(axis=0, loc_start=4)  # loc_start exceeds end index
        # if this line is reached, trim failed to raise an exception
        assert False
    except IndexError:
        # IndexError was raised (expected behavior)
        assert True
    try:
        slice.trim(axis=1, loc_start=2, loc_end=3)
        # if this line is reached, trim failed to raise an exception
        assert False
    except IndexError:
        # IndexError was raised (expected behavior)
        assert True


def test_shape():
    """Tests that `shape()` return the proper values."""
    # Define a data array and pixel_dimensions
    data = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ]
    )
    pixel_dimensions = [2.0, 4.0]

    # Create the slice object
    slice = Slice(data=data, pixel_dimensions=pixel_dimensions)
    # Assert the slice was created correctly
    assert slice.shape() == data.shape


def test_filter():
    """Tests the `filter` method on the `Slice` object."""
    # define the core
    shape: list[int] = [2, 4]
    data: np.ndarray = np.zeros(shape)
    counter: int = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            data[x, y] = counter
            counter += 1

    # define the slice object and filter function
    slice: Slice = Slice(data=data, pixel_dimensions=(2.0, 4.0))
    filter_func = lambda a: True if 3 <= a <= 5 else False  # noqa

    # filter the slice
    filtered_slice: Slice = slice.filter(filter_func)

    # ensure that new core is the same size
    assert filtered_slice.pixel_dimensions == slice.pixel_dimensions

    # ensure that values left are only the filtered values
    for i, row in enumerate(filtered_slice.data):
        for brightness in row:
            assert np.isnan(brightness) or 3 <= brightness <= 5
