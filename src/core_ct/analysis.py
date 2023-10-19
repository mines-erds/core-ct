"""Methods to quantitatively analyze CT scan data."""

import numpy as np
import pandas as pd


def brightness_trace(slice):
    """
    Compute the mean of the brightnesses and its standard deviation for each layer.

    Arguments:
    ---------
        slice: a slice of the core as a 3D numpy array

    Returns:
    -------
        A 2D numpy array where the first column is the mean of the brightness and
        the second is the standard deviation.
    """
    # Return a pandas dataframe containing the mean and STD for each slice row
    return pd.DataFrame(
        {"mean": np.mean(slice, axis=1), "stddev": np.std(slice, axis=1)}
    )
