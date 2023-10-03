import numpy as np
import pandas as pd

def brightness_trace(slice):
    """Takes a slice and will return the mean of the brightnesses along that core and the standard deviation for the
    same slice.

    :param slice: A slice of the core as a 3D numpy array
    :return: A 2D numpy array where the first column is the mean of the brightness and the second is the
            standard deviation.
    """
    # Return a pandas dataframe containing the mean and STD for each slice row
    return pd.DataFrame({
        'mean': np.mean(slice, axis=1),
        'stddev': np.std(slice, axis=1)
    })
