import numpy as np

def brightness_trace(slice):
    """
    Takes a slice and will return the mean of the brightnesses along that core and the standard deviation for the
    same slice.

    :param slice: A slice of the core as a 3D numpy array
    :return: A 2D numpy array where the first column is the mean of the brightness and the second is the
            standard deviation.
    """

    # Flip the two result arrays to make the orientation correct
    brightness_result = np.mean(slice, axis=1)
    std_result = np.std(slice, axis=1)

    # TODO: Return this as a pandas dataframe with labels
    # Make the two result arrays into one returnable array
    return np.array([brightness_result, std_result]).T
