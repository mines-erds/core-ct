"""Methods to quantitatively analyze CT scan data."""

import numpy as np
import pandas as pd
from core_ct.slice import Slice


def brightness_trace(slice: Slice) -> pd.DataFrame:
    """
    Compute the mean brightness across each row of a slice and its standard deviation.

    Arguments
    ---------
    slice : np.ndarray
        A slice of the core as a 3D numpy array

    Returns
    -------
    pd.DataFrame
        A 2D numpy array where the first column is the mean of the brightness and
        the second is the standard deviation.
    """
    # Return a pandas dataframe containing the mean and STD for each slice row
    return pd.DataFrame(
        {"mean": np.mean(slice.data, axis=1), "stddev": np.std(slice.data, axis=1)}
    )
