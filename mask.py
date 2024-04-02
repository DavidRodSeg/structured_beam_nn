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

    def Identity(self):
        """
        Applies an identity mask.
        :return: array of the identity mask
        """
        self.mask = np.ones((self.xdim, self.ydim), dtype=np.complex64)

        return self
    
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
    
    def setSlit(self, N, size=1):
        """
        Applies a mask that is equivalent of a N-slit (pure-amplitude mask with N apertures)
         with slits separate by the same length.
        :param N: number of apertures in the mask. The maximum number of slits depends on the
         size of the beam's matrix and the lenght of the slits
        :param size: size of the slits (in pixel size)
        :return: array of the beam with the mask applied
        """
        steps = int(( self.xdim - N*size ) / ( N+1 ))
        self.mask = np.zeros((self.xdim, self.ydim), dtype=np.complex64) # Zero mask = wall (the light doesn't goes through the mask)
        for j in range(steps+size, self.xdim, steps+size): # REVISAR EL CÓMO CENTRAR LA RENDIJA
            for i in range(size):
                self.mask[:,j+i-size] = 1
        
        return self
    
    def setRectangle(self, a, b):
        """
        Applies a rectangle mask (amplitude-only mask).
        :param a: horizontal side of the rectangle (in pixel size)
        :param b: vertical side of the rectangle (in pixel size)
        :return: array of the beam with the mask applied
        """
        self.mask = np.zeros((self.xdim, self.ydim), dtype=np.complex64)
        x0 = int((self.xdim - a)/2)
        y0 = int((self.ydim - b)/2)
        for i in range(x0, a + x0, 1):
            for j in range(y0, b + y0, 1):
                self.mask[i,j] = 1

        return self
    
    def setCircular(self, r):
        """
        Applies a circular mask (amplitude-only mask).
        :param r: radius of the aperture (in pixel size)
        :return: array of the beam with the mask applied
        """
        self.mask = np.zeros((self.xdim, self.ydim), dtype=np.complex64)
        for i in range(self.xdim):
            for j in range(self.ydim):
                if int(np.sqrt(( i-self.xdim/2 )**2 + ( j-self.ydim/2 )**2)) <= r: # The origin is established in the coordinate (xdim/2, ydim/2) which means is centered
                    self.mask[i,j] = 1
        
        return self

    def Apply(self, beam):
        """
        Apply the mask to the desired structured beam.
        :param abeam: array of the field's cross-sectional profile
        :return: array of the beam with the mask applied
        """
        product = np.multiply(self.mask, beam)

        return product
    
    def get(self):
        """
        Allow accesing to the mask's array.
        :return: mask's array
        """
        return self.mask