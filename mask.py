"""
Methods for creating and applying a mask to a an array representing a structured beam created by the scalar_monobeam module.
"""


# Import libraries
import numpy as np
from scipy.ndimage import rotate
import cv2


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
        self.mask = np.zeros((self.xdim, self.ydim), dtype=np.float32) # Zero mask = wall (the light doesn't goes through the mask)
        for j in range(steps+size, self.xdim, steps+size): # REVISAR EL CÓMO CENTRAR LA RENDIJA
            for i in range(size):
                self.mask[:,j+i-size] = 1
        
        return self
    
    def setRectangle(self, a, b):
        """
        Applies a rectangular mask (amplitude-only mask).
        :param a: horizontal side of the rectangle (in pixel size)
        :param b: vertical side of the rectangle (in pixel size)
        :return: array of the beam with the mask applied
        """
        self.mask = np.zeros((self.xdim, self.ydim), dtype=np.float32)
        x0 = int((self.xdim - a)/2)
        y0 = int((self.ydim - b)/2)
        for i in range(x0, a + x0, 1):
            for j in range(y0, b + y0, 1):
                self.mask[i,j] = 1

        return self
    
    def setCircular(self, r, random_center=False):
        """
        Applies a circular mask (amplitude-only mask).
        :param r: radius of the aperture (in pixel size)
        :return: array of the beam with the mask applied
        """
        self.mask = np.zeros((self.xdim, self.ydim), dtype=np.float32)
        if random_center == False:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    if int(np.sqrt(( i-self.xdim/2 )**2 + ( j-self.ydim/2 )**2)) <= r: # The origin is established in the coordinate (xdim/2, ydim/2) which means is centered
                        self.mask[i,j] = 1
        else:
            r1 = np.random.randint(-5, 5)
            r2 = np.random.randint(-5, 5)
            for i in range(self.xdim):
                for j in range(self.ydim):
                    if int(np.sqrt(( i-self.xdim/2 + r1 )**2 + ( j-self.ydim/2 + r2 )**2)) <= r:
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
        Allow acces to the mask's array.
        :return: mask's array
        """
        return self.mask
    
    def noise(self):
        """
        Generate and apply noise (in amplitude) to the mask.
        :return: mask's array
        """
        noise = np.random.uniform(0,5*10e-2, [self.xdim, self.ydim])
        self.mask = self.mask + noise

        return self
    
    def setTriangular(self, l, origin=False):
        """
        Applies a triangular (regular) mask (amplitude-only mask).
        :param l: side of the triangle (in pixel size)
        :return: array of the beam with the mask applied
        """
        self.mask = np.zeros((self.xdim, self.ydim), dtype=np.float32)
        m1 = np.tan(np.pi/3)
        m2 = - m1
        h = l * np.sqrt(3) / 2
        if origin == False:
            barycenter = l * np.sqrt(3) / 3
            originx = int(self.xdim/2)
            originy = int(self.xdim/2 - barycenter)
        elif origin == True:
            originx = int(self.xdim/2)
            originy = int(self.xdim/2)
        else:
            raise(NotImplementedError)

        for i in range(self.xdim):
            for j in range(self.ydim):
                condition1 = int(m1*(i-originx))+originy
                condition2 = int(m2*(i-originx))+originy
                condition3 = h + originy
                if (j > condition1 and j > condition2 and j < condition3):
                    self.mask[j,i] = 1

        return self
        
    def rotation(self, alpha):
        """
        Rotate the mask.
        :param alpha: angle of rotation in degrees
        :return: array with the rotation applied
        """

        self.mask = rotate(self.mask, alpha, reshape=False)

        return self

    def setRegularPolygon(self, radius, num_sides, center=None, rotation_angle=0):
        """
        Applies a regular polygon mask (amplitude-only mask).
        :param radius: radius of the polygon (in pixel size)
        :param num_sides: number of sides of the polygon
        :param center: (x,y) tuple with the shift of the coordinates of the center
         with respect to the image center (in pixel size)
        :param rotation_angle: angle of the rotation applied to the polygon
        :return: array of the beam with the mask applied
        """
        vertices = []
        angle_step = 2 * np.pi / num_sides

        if center == None:
            center = [self.xdim//2, self.ydim//2]
        else:
            center = [self.xdim//2 + center[0], self.ydim//2 + center[1]]

        for i in range(num_sides):
            theta = i * angle_step + rotation_angle
            x = int(center[0] + radius * np.cos(theta))
            y = int(center[1] + radius * np.sin(theta))
            vertices.append([x, y])

        vertices_array = np.array(vertices, np.int32)
        mask = np.zeros((self.xdim, self.ydim), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices_array], 1)
        self.mask = mask

        return self