"""Tests the `Core` class in the `core` module."""

from core_ct.core import Core
import numpy as np
import copy


def test_core():
    """Tests that a `Core` object can be created successfully."""
    # Define the pixel array and pixel_dimensions
    pixel_array = np.zeros([2, 4, 8])
    pixel_dimensions = [2.0, 4.0, 8.0]

    # Create the core object
    core = Core(pixel_array=pixel_array, pixel_dimensions=pixel_dimensions)

    # Assert the core was created correctly
    np.testing.assert_equal(core.pixel_array, pixel_array)
    np.testing.assert_equal(core.pixel_dimensions, pixel_dimensions)


def test_slice():
    """Tests the `slice` method on the `Core`."""
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


def test_swapaxes():
    """Tests the `slice` method on the `Core`."""
    # Define the core
    shape: list[int] = [2, 4, 8]
    pixel_array: np.ndarray = np.zeros(shape)
    counter: int = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                pixel_array[x, y, z] = counter
                counter += 1

    core: Core = Core(
        pixel_array=copy.deepcopy(pixel_array), pixel_dimensions=[2.0, 4.0, 8.0]
    )

    # Swap various axes
    core_xy: Core = core.swapaxes(0, 1)
    core_xz: Core = core.swapaxes(0, 2)
    core_yz: Core = core.swapaxes(1, 2)

    # Check that the swap operation worked correctly on pixel_array
    swap_xy: np.ndarray = np.swapaxes(pixel_array, 0, 1)
    assert core_xy.pixel_array.shape == swap_xy.shape
    assert np.array_equal(core_xy.pixel_array, swap_xy)
    swap_xz: np.ndarray = np.swapaxes(pixel_array, 0, 2)
    assert core_xz.pixel_array.shape == swap_xz.shape
    assert np.array_equal(core_xz.pixel_array, swap_xz)
    swap_yz: np.ndarray = np.swapaxes(pixel_array, 1, 2)
    assert core_yz.pixel_array.shape == swap_yz.shape
    assert np.array_equal(core_yz.pixel_array, swap_yz)

    # Check that pixel_dimensions were updated correctly
    assert core_xy.pixel_dimensions == [4.0, 2.0, 8.0]
    assert core_xz.pixel_dimensions == [8.0, 4.0, 2.0]
    assert core_yz.pixel_dimensions == [2.0, 8.0, 4.0]

    # Check that Core.swapaxes() raises exceptions correctly
    try:
        core.swapaxes(-1, 0)
        # if this line is reached, swapaxes failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.pixel_array, pixel_array)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == [2.0, 4.0, 8.0]
    try:
        core.swapaxes(0, -1)
        # if this line is reached, swapaxes failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.pixel_array, pixel_array)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == [2.0, 4.0, 8.0]
    try:
        core.swapaxes(3, 0)
        # if this line is reached, swapaxes failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.pixel_array, pixel_array)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == [2.0, 4.0, 8.0]
    try:
        core.swapaxes(0, 3)
        # if this line is reached, swapaxes failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.pixel_array, pixel_array)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == [2.0, 4.0, 8.0]


def test_flip():
    """Tests the `flip` method on the `Core`."""
    # Define the core
    shape: list[int] = [2, 4, 8]
    pixel_array: np.ndarray = np.zeros(shape)
    counter: int = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                pixel_array[x, y, z] = counter
                counter += 1

    core: Core = Core(
        pixel_array=copy.deepcopy(pixel_array), pixel_dimensions=[2.0, 4.0, 8.0]
    )

    # Flip various axes
    core_x: Core = core.flip(0)
    core_y: Core = core.flip(1)
    core_z: Core = core.flip(2)

    # Check that the flip operation worked correctly on pixel_array
    flip_x: np.ndarray = np.flip(pixel_array, 0)
    assert core_x.pixel_array.shape == flip_x.shape
    assert np.array_equal(core_x.pixel_array, flip_x)
    flip_y: np.ndarray = np.flip(pixel_array, 1)
    assert core_y.pixel_array.shape == flip_y.shape
    assert np.array_equal(core_y.pixel_array, flip_y)
    flip_z: np.ndarray = np.flip(pixel_array, 2)
    assert core_z.pixel_array.shape == flip_z.shape
    assert np.array_equal(core_z.pixel_array, flip_z)

    # Check that pixel_dimensions remained constant
    assert core_x.pixel_dimensions == shape
    assert core_y.pixel_dimensions == shape
    assert core_z.pixel_dimensions == shape

    # Check that Core.rotate() raises exceptions correctly
    try:
        core.flip(-1)
        # if this line is reached, rotate failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.pixel_array, pixel_array)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == [2.0, 4.0, 8.0]
    try:
        core.flip(3)
        # if this line is reached, rotate failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.pixel_array, pixel_array)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == [2.0, 4.0, 8.0]


def test_rotate():
    """Tests the `rotate` method on the `Core`."""
    # Define the core
    shape: list[int] = [2, 4, 8]
    pixel_array: np.ndarray = np.zeros(shape)
    counter: int = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                pixel_array[x, y, z] = counter
                counter += 1

    core: Core = Core(
        pixel_array=copy.deepcopy(pixel_array), pixel_dimensions=[2.0, 4.0, 8.0]
    )

    # Rotate various axes
    core_x: Core = core.rotate(0, k=1)
    core_y: Core = core.rotate(1, k=1)
    core_z: Core = core.rotate(2, k=1)

    # Check that the flip operation worked correctly on pixel_array
    rot_x: np.ndarray = np.rot90(pixel_array, k=1, axes=(1, 2))
    assert core_x.pixel_array.shape == rot_x.shape
    assert np.array_equal(core_x.pixel_array, rot_x)
    rot_y: np.ndarray = np.rot90(pixel_array, k=1, axes=(0, 2))
    assert core_y.pixel_array.shape == rot_y.shape
    assert np.array_equal(core_y.pixel_array, rot_y)
    rot_z: np.ndarray = np.rot90(pixel_array, k=1, axes=(0, 1))
    assert core_z.pixel_array.shape == rot_z.shape
    assert np.array_equal(core_z.pixel_array, rot_z)

    # Check that pixel_dimensions were updated correctly
    assert core_x.pixel_dimensions == [2.0, 8.0, 4.0]
    assert core_y.pixel_dimensions == [8.0, 4.0, 2.0]
    assert core_z.pixel_dimensions == [4.0, 2.0, 8.0]

    # Rotate various axes with an even k
    core_x: Core = core.rotate(0, k=2)
    core_y: Core = core.rotate(1, k=2)
    core_z: Core = core.rotate(2, k=2)

    # Check that the flip operation worked correctly on pixel_array
    rot_x: np.ndarray = np.rot90(pixel_array, k=2, axes=(1, 2))
    assert core_x.pixel_array.shape == rot_x.shape
    assert np.array_equal(core_x.pixel_array, rot_x)
    rot_y: np.ndarray = np.rot90(pixel_array, k=2, axes=(0, 2))
    assert core_y.pixel_array.shape == rot_y.shape
    assert np.array_equal(core_y.pixel_array, rot_y)
    rot_z: np.ndarray = np.rot90(pixel_array, k=2, axes=(0, 1))
    assert core_z.pixel_array.shape == rot_z.shape
    assert np.array_equal(core_z.pixel_array, rot_z)

    # Check that pixel_dimensions were not changed
    assert core_x.pixel_dimensions == shape
    assert core_y.pixel_dimensions == shape
    assert core_z.pixel_dimensions == shape

    # Rotate various axes clockwise
    core_x: Core = core.rotate(0, k=1, clockwise=True)
    core_y: Core = core.rotate(1, k=1, clockwise=True)
    core_z: Core = core.rotate(2, k=1, clockwise=True)

    # Check that the flip operation worked correctly on pixel_array
    rot_x: np.ndarray = np.rot90(pixel_array, k=-1, axes=(1, 2))
    assert core_x.pixel_array.shape == rot_x.shape
    assert np.array_equal(core_x.pixel_array, rot_x)
    rot_y: np.ndarray = np.rot90(pixel_array, k=-1, axes=(0, 2))
    assert core_y.pixel_array.shape == rot_y.shape
    assert np.array_equal(core_y.pixel_array, rot_y)
    rot_z: np.ndarray = np.rot90(pixel_array, k=-1, axes=(0, 1))
    assert core_z.pixel_array.shape == rot_z.shape
    assert np.array_equal(core_z.pixel_array, rot_z)

    # Check that pixel_dimensions were updated correctly
    assert core_x.pixel_dimensions == [2.0, 8.0, 4.0]
    assert core_y.pixel_dimensions == [8.0, 4.0, 2.0]
    assert core_z.pixel_dimensions == [4.0, 2.0, 8.0]

    # Check that Core.rotate() raises exceptions correctly
    try:
        core.rotate(-1)
        # if this line is reached, rotate failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.pixel_array, pixel_array)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == [2.0, 4.0, 8.0]
    try:
        core.rotate(3)
        # if this line is reached, rotate failed to raise an exception
        assert False
    except ValueError:
        # ValueError was raised (expected behavior), make sure no data was altered
        assert np.array_equal(core.pixel_array, pixel_array)
        # Make sure pixel_dimensions weren't altered
        assert core.pixel_dimensions == [2.0, 4.0, 8.0]


def test_join():
    """Tests the `join` method on the `Core`."""
    # Define the target core for joining
    target = Core(np.zeros([2, 4, 8]), [2.0, 4.0, 8.0])
    # Define the source cores for joining
    source_valid = Core(np.zeros([2, 4, 8]), [2.0, 4.0, 8.0])
    # source_invalid_dimensions = Core(np.zeros([2, 4, 8]), [2.0, 4.0, 8.0])
    # source_invalid_shape = Core(np.zeros([2, 4, 8]), [2.0, 4.0, 8.0])

    # Assert the valid core joins together properly on each axis
    joined_valid_0 = target.join(source_valid, axis=0)
    # joined_valid_1 = target.join(source_valid, axis=1)
    # joined_valid_2 = target.join(source_valid, axis=2)
    assert joined_valid_0.pixel_array.shape == (4, 4, 8)

    # Test that the join method fails on invalid dimensions
