import numpy as np
from tqdm import tqdm

c = 3*10**8

class Fraunhofer:
    def __init__(self, samples, f=1000, z=0):
        """
        This class implements all the procedures necessary for implementing a 2D Discrete Fourier
         Transformation for a given samples data.
        :param samples: matrix of the samples that will be transformed
        """
        self.samples = samples
        self.xdim = samples.shape[0]
        self.ydim = samples.shape[1]
        self.f = f
        self.z = z
        self.N = int(self.xdim/2)
        self.wavelength = c / self.f

    def DFT_1D(self, x, j=0):
        """
        Computes the Discrete Fourier Transformation for a 1D samples array.
        :param x: coordinate in cartesian coordinates
        :return: value of the DFT in the specified point
        """
        dft = 0.
        vx = self.__frequency(x)
        dx = 0.01
        for i in range(0, self.xdim, 1):
            dft += self.samples[i, j] * np.exp( -2j*np.pi*vx*i*dx )

        return dft

    def DFT_2D(self, x, y):
        """
        Computes the Discrete Fourier Transformation for a 2D samples array.
        :param x: x coordinate in cartesian coordinates
        :param y: y coordinate in cartesian coordinates 
        :return: value of the DFT in the specified points
        """
        dft = 0.
        vy = self.__frequency(y)
        dy = 0.01

        for j in range(0, self.ydim, 1):
            dft += self.DFT_1D(x, j) * np.exp( -2j*np.pi*vy*j*dy )

        return dft

    def __frequency(self, x):
        """
        Calculates the frequency in the specify coordinate.
        :param x: x coordinate in cartesian coordinates
        :return: value of the frequency in the x coordinate
        """
        vx = x / ( self.wavelength*self.z )

        return vx
    
    def diffraction(self):
        """
        Propagates the initial beam using the Fraunhofer diffraction equation
        :param z: z coordinate in cartesian coordinates. Distance to evaluate
         the propagation
        :return: array of xdim x ydim dimensions for the propagted beam
        """
        M = int(self.N/0.5)
        dif_arr = np.empty((2*M, 2*M), dtype=np.complex64)
        k = 2*np.pi/self.wavelength
        A = np.exp( 1j*k*self.z )/( 1j*self.wavelength*self.z )
        dx = 0.05
        dy = 0.05
        for i in tqdm(range(0, 2*M, 1)):
            for j in range(0, 2*M, 1):
                dif_arr[i,j] = A*self.DFT_2D(x=dx*(i-M), y=dy*(j-M))*dx*dy

        return dif_arr
