import numpy as np

class Analysis:
    @staticmethod
    def brightness_trace(slice):
        # Initialize the result variables to store data in
        brightnessResult = np.zeros(0)
        STDResult = np.zeros(0)

        # Step through each section of the slice that was passed in
        for section in slice:
            # Calculate the average brightness of that section and add it to the result array
            brightnessOfSlice = section.sum()
            brightnessOfSlice /= section.size
            brightnessResult = np.append(brightnessOfSlice, brightnessResult)

            # Calculate the standard deviation of that slice and add it to the result array
            STDOfSlice = section.std()
            STDResult = np.append(STDOfSlice, STDResult)
        # Flip the two result arrays to make the orientation correct
        brightnessResult = np.flip(brightnessResult)
        STDResult = np.flip(STDResult)

        # Make the two result arrays into one returnable array
        result = np.vstack((brightnessResult, STDResult))

        return result
