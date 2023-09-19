from core_ct import importers
from os.path import dirname, realpath, join

# Get the test directory path
tests_dir = dirname(realpath(__file__))
# Get the core directory path
core_dir = join(tests_dir, 'scans', 'PAT_4615_18L_0001')

# Import the core from the DICOM files
core = importers.dicom(core_dir)

print(core.pixel_array)
