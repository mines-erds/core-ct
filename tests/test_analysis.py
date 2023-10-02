from core_ct import analysis
import numpy as np

def test_brightness_trace():
    # Define a core "slice"
    slice = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
    ])

    # Perform a brightness trace of the "slice"
    bt = analysis.brightness_trace(slice)

    # Check that the mean of each row was taken correctly
    assert bt['mean'][0] == np.mean(slice[0])
    assert bt['mean'][1] == np.mean(slice[1])
    assert bt['mean'][2] == np.mean(slice[2])

    # Check that the STD of each row was taken correctly
    assert bt['std'][0] == np.std(slice[0])
    assert bt['std'][1] == np.std(slice[1])
    assert bt['std'][2] == np.std(slice[2]) 
