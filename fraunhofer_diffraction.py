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

        self.dx = 0.01
        self.dy = 0.01
        self.dx2 = 0.01
        self.dy2 = 0.01
        self.A = ( self.xdim*self.dx*self.dx2 )/ ( self.wavelength*self.z )
        self.amp = 2

    def DFT_1D(self, x, j=0):
        """
        Computes the Discrete Fourier Transformation for a 1D samples array in a single
         point (x,).
        :param x: coordinate in cartesian coordinates
        :return: value of the DFT in the specified point
        """
        dft = 0.
        vx = self.__frequency(x)
        for i in range(0, self.xdim, 1):
            dft += self.samples[i, j] * np.exp( -2j*np.pi*vx*i*self.dx )

        return dft

    def DFT_2D(self, x, y):
        """
        Computes the Discrete Fourier Transformation for a 2D samples array in a single
         point (x, y).
        :param x: x coordinate in cartesian coordinates
        :param y: y coordinate in cartesian coordinates 
        :return: value of the DFT in the specified points
        """
        dft = 0.
        vy = self.__frequency(y)

        for j in range(0, self.ydim, 1):
            dft += self.DFT_1D(x, j) * np.exp( -2j*np.pi*vy*j*self.dy )

        return dft

    def __frequency(self, x):
        """
        Calculates the frequency in the specify coordinate.
        :param x: x coordinate in cartesian coordinates
        :return: value of the frequency in the x coordinate
        """
        vx = x / ( self.wavelength*self.z )

        return vx

    def __SFT_1D(self, x):
        """
        Appllies the Slow (normal) Fourier Trasnformation in 1D using numpy arrays.
        :return: value of the DFT in the specified points
        """
        N = x.shape[0]
        M = int(self.amp*N/2)
        n = np.arange(-N/2, N/2, 1)
        k = np.reshape(np.arange(-M, M, 1), (2*M,1)) # Vertical vector to be able to create the M(m x n) array. We defined k from -M to M to get a zero-centered image
        M = np.exp( -2j*np.pi*self.A*k*n / N )

        return np.dot(M, x)

    def FFT_1D(self, x):
        """
        Appllies the Fast Fourier Trasnformation in 1D. Implementation of
         the Cooley-Turkey algorithm.
        :return: value of the DFT in the specified points
        """
        x = np.asarray(x, dtype=np.complex64)
        N = x.shape[0]
        M = int(self.amp*N/2)
        
        if N % 2 > 0:
            raise ValueError("X dimension must be a power of 2")
        elif N <= 2*self.xdim: # The cutoff should be optimized (depending on the value it could produce 2N images of the results)
            return self.__SFT_1D(x)
        else:
            x_even = self.FFT_1D(x[::2]) # We take the even indices
            x_odd = self.FFT_1D(x[1::2]) # Odd indices
            factor = np.exp( -2j*np.pi*self.A*np.arange(-M, M, 1) / (N/2) ) # REVISAR SI ESTÁN BIEN LOS FACTORES DE /2
            return np.concatenate([x_even + factor[:M]*x_odd, x_even + factor[M:]*x_odd]) # ANTES SE TOMABA factor[:M//2]
        
    def FFT_2D(self, x):
        """
        Appllies the Fast Fourier Trasnformation in 2D. Implementation of
         the Cooley-Turkey algorithm.
        :return: value of the DFT in the specified points
        """
        fft = np.empty((self.amp*x.shape[0], self.amp*x.shape[1]), dtype=np.complex64)
        y = np.empty((self.amp*x.shape[0], x.shape[1]), dtype=np.complex64)
        x = np.asarray(x, dtype=np.complex64)
        for j in range(x.shape[1]):
            y[:,j] = self.FFT_1D(x[:,j])
        for i in range(self.amp*x.shape[0]):
            fft[i,:] = self.FFT_1D(y[i,:])
        return fft
       
    def diffraction(self):
        """
        Propagates the initial beam using the Fraunhofer diffraction equation
        :param z: z coordinate in cartesian coordinates. Distance to evaluate
         the propagation
        :return: array of xdim x ydim dimensions for the propagted beam
        """
        M = int(self.amp*self.N)
        dif_arr = np.empty((M, M), dtype=np.complex64)
        k = 2*np.pi/self.wavelength
        A = np.exp( 1j*k*self.z )/( 1j*self.wavelength*self.z )
        # for i in tqdm(range(0, 2*M, 1)):
        #     for j in range(0, 2*M, 1):
        #         dif_arr[i,j] = A*self.DFT_2D(x=self.dx*(i-M), y=self.dy*(j-M))*self.dx2*self.dy2
        dif_arr = A*self.FFT_2D(self.samples)*self.dx2*self.dy2

        return dif_arr
            