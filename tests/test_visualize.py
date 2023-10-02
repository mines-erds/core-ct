from core_ct import importers, visualize
import os

# Define the paths for the tests and scans directory
tests_dir = os.path.dirname(os.path.realpath(__file__))
scans_dir = os.path.join(tests_dir, 'scans')

def test_slice():
    # Import a core for testing
    scan_dir = os.path.join(scans_dir, 'PAT_4636_39L_0001')
    core = importers.dicom(dir=scan_dir)

    # Define the location of the image
    test_file = os.path.join(tests_dir, 'test_visualize_slice.png')

    # Delete the file if it already exists
    if os.path.isfile(test_file):
        os.remove(test_file)

    # Create the image
    visualize.slice(core, test_file, 2, 0)

    # Assert that the file was created
    assert os.path.isfile(test_file)

    # Delete the file
    os.remove(test_file)

    # Assert that the file was deleted
    assert not os.path.isfile(test_file)
