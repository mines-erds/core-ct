"""Tests the `Core` class in the `core` module."""

from core_ct.core import Core
from core_ct import importers
import numpy as np
import os
import pytest


tests_dir = os.path.dirname(os.path.realpath(__file__))
scans_dir = os.path.join(tests_dir, "scans")


def test_core():
    """Tests that a `Core` object can be created successfully."""
    # Define the data array and pixel_dimensions
    data = np.zeros([2, 4, 8])
    pixel_dimensions = [2.0, 4.0, 8.0]

    # Create the core object
    core = Core(data=data, pixel_dimensions=pixel_dimensions)

    # Assert the core was created correctly
    np.testing.assert_equal(core.data, data)
    np.testing.assert_equal(core.pixel_dimensions, pixel_dimensions)


def test_slice():
    """Tests the `slice` method on the `Core`."""
    # Define the core parameters
    data = np.array(
        [
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]],
            [[31, 32, 33, 34, 35], [36, 37, 38, 39, 40], [41, 42, 43, 44, 45]],
            [[46, 47, 48, 49, 50], [51, 52, 53, 54, 55], [56, 57, 58, 59, 60]],
        ]
    )
    pixel_dimensions = (2.0, 4.0, 8.0)

    # Create the Core object
    core = Core(data=data, pixel_dimensions=pixel_dimensions)

    # Take slice from each axis out of the core
    slice_0 = core.slice(axis=0, loc=0)
    slice_1 = core.slice(axis=1, loc=0)
    slice_2 = core.slice(axis=2, loc=0)

    # Check that the shape of each slice is correct
    assert slice_0.shape() == (3, 5)
    assert slice_1.shape() == (4, 5)
    assert slice_2.shape() == (4, 3)

    # Check that the data values are as expected
    assert np.array_equal(
        slice_0.data,
        np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]),
    )
    assert np.array_equal(
        slice_1.data,
        np.array(
            [
                [1, 2, 3, 4, 5],
                [16, 17, 18, 19, 20],
                [31, 32, 33, 34, 35],
                [46, 47, 48, 49, 50],
            ]
        ),
    )
    assert np.array_equal(
        slice_2.data, np.array([[1, 6, 11], [16, 21, 26], [31, 36, 41], [46, 51, 56]])
    )
    # assert that the pixel dimensions are correct
    assert slice_0.pixel_dimensions == (4.0, 8.0)
    assert slice_1.pixel_dimensions == (2.0, 8.0)
    assert slice_2.pixel_dimensions == (2.0, 4.0)

def test_trim_radial():
    """Tests the 'trim_radial` method on the `Core`."""
    # Define a simple Core
    shape: list[int] = (9, 9, 16)
    data: np.ndarray = np.zeros(shape)
    counter: int = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                data[x, y, z] = counter
                counter += 1

    core: Core = Core(
        data=data, pixel_dimensions=(1.0, 1.0, 1.0)
    )

    trimmed: Core = core.trim_radial(axis=2, radius=3.0, x_center=4, y_center=4)

    # make sure the matrix was reduced correctly
    assert trimmed.shape() == (7, 7, 16)

    # make sure corners are empty
    assert np.isnan(trimmed.data[0,0,0])
    assert np.isnan(trimmed.data[0,6,0])
    assert np.isnan(trimmed.data[6,6,0])
    assert np.isnan(trimmed.data[6,0,0])

    # make sure it is inclusive if distance == radius
    assert not np.isnan(trimmed.data[3,0,0])
    assert not np.isnan(trimmed.data[6,3,0])
    assert not np.isnan(trimmed.data[3,6,0])
    assert not np.isnan(trimmed.data[0,3,0])

    # make sure correct data was maintained
    assert np.array_equal(trimmed.data[1:6, 1:6, :], data[2:7, 2:7, :])

    # test trimming where radius includes entire Core
    trimmed = core.trim_radial(axis=2, radius=100.0, x_center=4, y_center=4)

    # make sure the matrix shape didn't change
    assert trimmed.shape() == (9, 9, 16)

    # make sure data wasn't changed
    assert np.array_equal(trimmed.data, data)

    # test trimming on a Core with irregular pixel/voxel dimensions
    core_irregular: Core = Core(
        data=data, pixel_dimensions=(1.0, 2.0, 4.0)
    )

    trimmed_irregular: Core = core_irregular.trim_radial(axis=2, radius=3.0, 
                                                         x_center=4, y_center=4)

    # make sure the matrix was reduced correctly
    assert trimmed_irregular.shape() == (7, 3, 16)

    # make sure corners are empty
    assert np.isnan(trimmed_irregular.data[0,0,0])
    assert np.isnan(trimmed_irregular.data[0,2,0])
    assert np.isnan(trimmed_irregular.data[6,2,0])
    assert np.isnan(trimmed_irregular.data[6,0,0])

    # make sure correct data was maintained
    assert np.array_equal(trimmed_irregular.data[1:6, 0:3, :], 
                          data[2:7, 3:6, :])

def test_swapaxes():
    """Tests the `slice` method on the `Core`."""
    # Define the core
    shape: list[int] = (2, 4, 8)
    data: np.ndarray = np.zeros(shape)
    counter: int = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                data[x, y, z] = counter
                counter += 1

    core: Core = Core(
        data=data, pixel_dimensions=(2.0, 4.0, 8.0)
    )

    # Swap various axes
    core_xy: Core = core.swapaxes(0, 1)
    core_xz: Core = core.swapaxes(0, 2)
    core_yz: Core = core.swapaxes(1, 2)

    # Check that the swap operation worked correctly on data
    swap_xy: np.ndarray = np.swapaxes(data, 0, 1)
    assert core_xy.data.shape == swap_xy.shape
    assert np.array_equal(core_xy.data, swap_xy)
    swap_xz: np.ndarray = np.swapaxes(data, 0, 2)
    assert core_xz.data.shape == swap_xz.shape
    assert np.array_equal(core_xz.data, swap_xz)
    swap_yz: np.ndarray = np.swapaxes(data, 1, 2)
    assert core_yz.data.shape == swap_yz.shape
    assert np.array_equal(core_yz.data, swap_yz)

    # Check that pixel_dimensions were updated correctly
    assert core_xy.pixel_dimensions == (4.0, 2.0, 8.0)
    assert core_xz.pixel_dimensions == (8.0, 4.0, 2.0)
    assert core_yz.pixel_dimensions == (2.0, 8.0, 4.0)

    # Check that Core.swapaxes() raises exceptions correctly
    try:
        core.swapaxes(-1, 0)
        # if this line is reached, swapaxes failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.data, data)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == (2.0, 4.0, 8.0)
    try:
        core.swapaxes(0, -1)
        # if this line is reached, swapaxes failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.data, data)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == (2.0, 4.0, 8.0)
    try:
        core.swapaxes(3, 0)
        # if this line is reached, swapaxes failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.data, data)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == (2.0, 4.0, 8.0)
    try:
        core.swapaxes(0, 3)
        # if this line is reached, swapaxes failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.data, data)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == (2.0, 4.0, 8.0)


def test_flip():
    """Tests the `flip` method on the `Core`."""
    # Define the core
    shape: list[int] = [2, 4, 8]
    data: np.ndarray = np.zeros(shape)
    counter: int = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                data[x, y, z] = counter
                counter += 1

    core: Core = Core(
        data=data, pixel_dimensions=(2.0, 4.0, 8.0)
    )

    # Flip various axes
    core_x: Core = core.flip(0)
    core_y: Core = core.flip(1)
    core_z: Core = core.flip(2)

    # Check that the flip operation worked correctly on data
    flip_x: np.ndarray = np.flip(data, 0)
    assert core_x.data.shape == flip_x.shape
    assert np.array_equal(core_x.data, flip_x)
    flip_y: np.ndarray = np.flip(data, 1)
    assert core_y.data.shape == flip_y.shape
    assert np.array_equal(core_y.data, flip_y)
    flip_z: np.ndarray = np.flip(data, 2)
    assert core_z.data.shape == flip_z.shape
    assert np.array_equal(core_z.data, flip_z)

    # Check that pixel_dimensions remained constant
    assert core_x.pixel_dimensions == (2.0, 4.0, 8.0)
    assert core_y.pixel_dimensions == (2.0, 4.0, 8.0)
    assert core_z.pixel_dimensions == (2.0, 4.0, 8.0)

    # Check that Core.rotate() raises exceptions correctly
    try:
        core.flip(-1)
        # if this line is reached, rotate failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.data, data)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == (2.0, 4.0, 8.0)
    try:
        core.flip(3)
        # if this line is reached, rotate failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.data, data)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == (2.0, 4.0, 8.0)


def test_rotate():
    """Tests the `rotate` method on the `Core`."""
    # Define the core
    shape: list[int] = [2, 4, 8]
    data: np.ndarray = np.zeros(shape)
    counter: int = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                data[x, y, z] = counter
                counter += 1

    core: Core = Core(
        data=data, pixel_dimensions=(2.0, 4.0, 8.0)
    )

    # Rotate various axes
    core_x: Core = core.rotate(0, k=1)
    core_y: Core = core.rotate(1, k=1)
    core_z: Core = core.rotate(2, k=1)

    # Check that the flip operation worked correctly on data
    rot_x: np.ndarray = np.rot90(data, k=1, axes=(1, 2))
    assert core_x.data.shape == rot_x.shape
    assert np.array_equal(core_x.data, rot_x)
    rot_y: np.ndarray = np.rot90(data, k=1, axes=(0, 2))
    assert core_y.data.shape == rot_y.shape
    assert np.array_equal(core_y.data, rot_y)
    rot_z: np.ndarray = np.rot90(data, k=1, axes=(0, 1))
    assert core_z.data.shape == rot_z.shape
    assert np.array_equal(core_z.data, rot_z)

    # Check that pixel_dimensions were updated correctly
    assert core_x.pixel_dimensions == (2.0, 8.0, 4.0)
    assert core_y.pixel_dimensions == (8.0, 4.0, 2.0)
    assert core_z.pixel_dimensions == (4.0, 2.0, 8.0)

    # Rotate various axes with an even k
    core_x: Core = core.rotate(0, k=2)
    core_y: Core = core.rotate(1, k=2)
    core_z: Core = core.rotate(2, k=2)

    # Check that the flip operation worked correctly on data
    rot_x: np.ndarray = np.rot90(data, k=2, axes=(1, 2))
    assert core_x.data.shape == rot_x.shape
    assert np.array_equal(core_x.data, rot_x)
    rot_y: np.ndarray = np.rot90(data, k=2, axes=(0, 2))
    assert core_y.data.shape == rot_y.shape
    assert np.array_equal(core_y.data, rot_y)
    rot_z: np.ndarray = np.rot90(data, k=2, axes=(0, 1))
    assert core_z.data.shape == rot_z.shape
    assert np.array_equal(core_z.data, rot_z)

    # Check that pixel_dimensions were not changed
    assert core_x.pixel_dimensions == (2.0, 4.0, 8.0)
    assert core_y.pixel_dimensions == (2.0, 4.0, 8.0)
    assert core_z.pixel_dimensions == (2.0, 4.0, 8.0)

    # Rotate various axes clockwise
    core_x: Core = core.rotate(0, k=1, clockwise=True)
    core_y: Core = core.rotate(1, k=1, clockwise=True)
    core_z: Core = core.rotate(2, k=1, clockwise=True)

    # Check that the flip operation worked correctly on data
    rot_x: np.ndarray = np.rot90(data, k=-1, axes=(1, 2))
    assert core_x.data.shape == rot_x.shape
    assert np.array_equal(core_x.data, rot_x)
    rot_y: np.ndarray = np.rot90(data, k=-1, axes=(0, 2))
    assert core_y.data.shape == rot_y.shape
    assert np.array_equal(core_y.data, rot_y)
    rot_z: np.ndarray = np.rot90(data, k=-1, axes=(0, 1))
    assert core_z.data.shape == rot_z.shape
    assert np.array_equal(core_z.data, rot_z)

    # Check that pixel_dimensions were updated correctly
    assert core_x.pixel_dimensions == (2.0, 8.0, 4.0)
    assert core_y.pixel_dimensions == (8.0, 4.0, 2.0)
    assert core_z.pixel_dimensions == (4.0, 2.0, 8.0)

    # Check that Core.rotate() raises exceptions correctly
    try:
        core.rotate(-1)
        # if this line is reached, rotate failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.data, data)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == (2.0, 4.0, 8.0)
    try:
        core.rotate(3)
        # if this line is reached, rotate failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.data, data)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == (2.0, 4.0, 8.0)


def test_filter():
    """Tests the `filter` method on the `Core`."""
    # Define the core
    shape: list[int] = [2, 4, 8]
    data: np.ndarray = np.zeros(shape)
    counter: int = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                data[x, y, z] = counter
                counter += 1

    core: Core = Core(data=data,
                      pixel_dimensions=(2.0, 4.0, 8.0))

    filter_func = lambda a: True if 3 <= a <= 8 else False  # noqa

    filtered_core: Core = core.filter(filter_func)

    # Ensure that new core is the same size
    assert filtered_core.pixel_dimensions == core.pixel_dimensions

    # ensure that values left are only the filtered values
    for i, row in enumerate(filtered_core.data):
        for j, col in enumerate(row):
            for brightness in col:
                assert np.isnan(brightness) or 3 <= brightness <= 8


def test_shape():
    """Tests the `shape` method on the `Core`."""
    # Create a couple of test cores
    core_0 = Core(np.zeros([2, 4, 8]), (2.0, 4.0, 8.0))
    core_1 = Core(np.zeros([5, 2, 6]), (2.0, 4.0, 8.0))
    core_2 = Core(np.zeros([9, 1, 7]), (2.0, 4.0, 8.0))

    # Assert that their shapes are correct
    assert core_0.shape() == (2, 4, 8)
    assert core_1.shape() == (5, 2, 6)
    assert core_2.shape() == (9, 1, 7)


def test_dimensions():
    """Tests the `dimensions` method on the `Core`."""
    # Create a couple of fake cores
    core_fake_0 = Core(np.zeros([2, 4, 8]), (2.0, 4.0, 8.0))
    core_fake_1 = Core(np.zeros([6, 1, 11]), (2.0, 4.0, 8.0))
    core_fake_2 = Core(np.zeros([2, 4, 8]), (0.31, 0.56, 1.2))

    # Assert that the size of the cores is correct
    assert core_fake_0.dimensions() == (4.0, 16.0, 64.0)
    assert core_fake_1.dimensions() == (12.0, 4.0, 88.0)
    assert core_fake_2.dimensions() == (2 * 0.31, 4 * 0.56, 8 * 1.2)

    # Import the scan from the directory
    core = importers.dicom(dir=os.path.join(scans_dir, "PAT_4636_39L_0001"))

    # Assert that the core imported correctly
    assert core.data.shape == (512, 512, 11)
    assert core.pixel_dimensions == (0.43, 0.43, 0.5)

    # Check that the dimensions of the core are correct
    assert core.dimensions() == (512 * 0.43, 512 * 0.43, 11 * 0.5)


def test_volume():
    """Tests the `volume` method on the `Core`."""
    # Define the data array and dimensions for the core
    shape = [2, 4, 8]
    data = np.zeros(shape)
    pixel_dimensions = (2.0, 4.0, 8.0)

    # Fill in the data array
    counter = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                data[x, y, z] = counter
                counter += 1

    # Define the voxel dimensions
    voxel_dimensions = pixel_dimensions[0] * pixel_dimensions[1] * pixel_dimensions[2]

    # Create the core from the data array and dimensions
    core = Core(data, pixel_dimensions)

    # TODO: Update these tests to use the volume function once it gets merged in
    # Verify the volume is correct within various density ranges
    assert core.volume() == 64 * voxel_dimensions


def test_join():
    """Tests the `join` method on the `Core`."""
    # Define the target core for joining
    target = Core(np.zeros([2, 4, 8]), (2.0, 4.0, 8.0))
    # Define the source cores for joining
    source_valid = Core(np.zeros([2, 4, 8]), (2.0, 4.0, 8.0))
    source_invalid_dimensions = Core(np.zeros([2, 4, 8]), (8.0, 4.0, 2.0))
    source_invalid_axis = Core(np.zeros([2, 16, 8]), (2.0, 4.0, 8.0))

    # Join the cores together on each axis
    joined_valid_0 = target.join(source_valid, axis=0)
    joined_valid_1 = target.join(source_valid, axis=1)
    joined_valid_2 = target.join(source_valid, axis=2)

    # Assert the valid core joins together properly on each axis
    assert joined_valid_0.data.shape == (4, 4, 8)
    assert joined_valid_1.data.shape == (2, 8, 8)
    assert joined_valid_2.data.shape == (2, 4, 16)

    # Test that a negative axis can't be passed
    with pytest.raises(ValueError):
        target.join(source_valid, axis=-1)

    # Test that an axis greater than 2 can't be passed
    with pytest.raises(ValueError):
        target.join(source_valid, axis=3)

    # Test that the join method fails on invalid dimensions
    with pytest.raises(
        ValueError,
        match=r".*\(8\.0, 4\.0, 2\.0\) != \(2\.0, 4\.0, 8\.0\)",
    ):
        target.join(source_invalid_dimensions, axis=0)

    # Test that the join method fails with an invalid shape along an axis
    with pytest.raises(ValueError):
        target.join(source_invalid_axis, axis=0)

    # Test that it works along another axis
    joined_invalid_axis = target.join(source_invalid_axis, axis=1)
    assert joined_invalid_axis.data.shape == (2, 20, 8)
