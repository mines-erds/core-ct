import numpy as np

class Core:
    """Core class contains properties of a core CT-scan and methods to isolate sections of the core

    Attributes:
        pixel_array -- 3D numpy array of pixel data that make up the core

    Methods:
        slice_2D(self, axis, loc) -- get a 2D slice of the core
    """

    def __init__(self, pixel_array):
        """Constructs necessary attributes of a Core

        Parameters:
            pixel_array - 3D array of pixel data that make up the core
        """
        # data must be in a numpy array for slicing methods to work
        if not isinstance(pixel_array, np.ndarray):
            self.pixel_array = np.array(pixel_array)
        else:
            self.pixel_array = pixel_array

    
    def slice_2D(self, axis, loc):
        """Get a single two-dimensional slice of the core scan at the specified location along the specified axis 
        A normal vector of the resulting plane will be parallel to the specified axis
        
        Parameters:
            axis -- integer either 0,1,2 specifying which dimension to collapse
            loc -- integer value along the axis specifying the location of the slice

        Returns:
            2D numpy array representing a single slice of the core
        """
        if axis not in [0,1,2]: 
            raise Exception("axis must be a value between 0 and 2 (inclusive)")
        
        if axis == 0:
            return self.pixel_array[loc]
        elif axis == 1:
            return self.pixel_array[:,loc]
        else: # axis == 2
            return self.pixel_array[:,:,loc]