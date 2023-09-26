
def test_core():
    core = Core(np.zeros([2, 4, 8]), [1.0, 1.0, 1.0])


import matplotlib.pyplot as plt
from os.path import dirname, realpath, join

# Get the test directory path
tests_dir = dirname(realpath(__file__))
# Get the core directory path
core_dir = join(tests_dir, 'scans', 'PAT_4636_39L_0001')

# Import the core from the DICOM files
core = importers.dicom(dir=core_dir)

# Get a slice of the core and manually trim it
slice = core.slice(axis=1, loc=10)
slice = slice.T[:, 200:300]

# Create a brightness trace of the graph
bt = analysis.brightness_trace(slice=slice)