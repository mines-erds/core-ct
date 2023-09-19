import numpy as np

def brightness_trace(slice):
    """
    Takes a slice and will return the mean of the brightnesses along that core and the standard deviation for the
    same slice.

    :param slice: A slice of the core as a 3D numpy array
    :return: A 2D numpy array where the first column is the mean of the brightness and the second is the
            standard deviation.
    """

    # Initialize the result variables to store data in
    brightness_result = np.zeros(0)
    std_result = np.zeros(0)

    # Step through each section of the slice that was passed in
    for section in slice:
        # Calculate the average brightness of that section and add it to the result array
        brightness_of_slice = section.sum()
        brightness_of_slice /= section.size
        brightness_result = np.append(brightness_of_slice, brightness_result)

        # Calculate the standard deviation of that slice and add it to the result array
        std_of_slice = section.std()
        std_result = np.append(std_of_slice, std_result)
    # Flip the two result arrays to make the orientation correct
    brightness_result = np.flip(brightness_result)
    std_result = np.flip(std_result)

    # Make the two result arrays into one returnable array
    result = np.vstack((brightness_result, std_result))

    return result
