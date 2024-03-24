"""
Methods for creating and applying a mask to a an array representing a structured beam created by the scalar_monobeam module.
"""



# Import libraries
import numpy as np



# Define the mask class
class Mask:
    def __init__(self, xdim, ydim):
        """
        Define a mask (array) with the dimensions specified.
        :param xdim: dimension of the x coordinate
        :param ydim: dimension of the y coordinate
        """
        self.xdim = xdim
        self.ydim = ydim
        self.mask = np.zeros((self.xdim, self.ydim), dtype=np.complex64) # By defect the mask is a wall. Transmission is not allowed.

    def setRandomMask(self):
        """
        Give random values to the mask.
        :return: numpy array of dimension (xdim, ydim) with random complex values
         with uniform distribution. Mask object for chaining.
        """
        self.mask = np.empty((self.xdim, self.ydim), dtype=np.complex64)
        for i in range((self.xdim)):
            for j in range((self.ydim)):
                self.mask[i,j] = np.sqrt(np.random.uniform()) * np.exp(1j*np.random.uniform(0, 2*np.pi))
        
        return self
    
    def setPeriodicMask(self):
        """
        """

    def Identity(self):
        """
        Applies an identity mask.
        :return: array of the identity mask
        """
        self.mask = np.ones((self.xdim, self.ydim), dtype=np.complex64)

        return self

    def Apply(self, beam):
        """
        Apply the mask to the desired structured beam.
        :param abeam: array of the field's cross-sectional profile
        :return: array of the beam with the mask applied
        """
        product = np.multiply(self.mask, beam)

        return product