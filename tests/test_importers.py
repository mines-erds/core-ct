"""Tests methods in the `importers` module."""

import os
from core_ct import importers
from glob import glob


# Define the paths for the tests and scans directory
tests_dir = os.path.dirname(os.path.realpath(__file__))
scans_dir = os.path.join(tests_dir, "scans")


def test_import_from_files():
    """Tests that the `dicom` method can successfully import from a list of files."""
    # Get the directory path for a specific scan
    scan_dir = os.path.join(scans_dir, "PAT_4636_39L_0001")
    # Construct a list of all the files for the scan
    scan_files = glob(scan_dir + "/PAT_4636_39L_*")

    # Assert that there are ten files
    assert len(scan_files) == 11

    # Import the scan from the file list
    core = importers.dicom(files=scan_files)

    # Assert that the core imported correctly
    assert core.pixel_array.shape == (512, 512, 11)


def test_import_from_dir():
    """Tests that the `dicom` method can successfully import from directory."""
    # Get the directory path for a specific scan
    scan_dir = os.path.join(scans_dir, "PAT_4636_39L_0001")

    # Import the scan from the directory
    core = importers.dicom(dir=scan_dir)

    # Assert that the core imported correctly
    assert core.pixel_array.shape == (512, 512, 11)
    assert core.pixel_dimensions == (0.43, 0.43, 0.5)
