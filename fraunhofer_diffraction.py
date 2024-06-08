import numpy as np

c = 3*10**8

class Fraunhofer:
    def __init__(self, samples, f=1000, z=0):
        """
        This class implements all the procedures necessary for implementing a 2D Discrete Fourier
         Transformation for a given samples data and the diffraction of a given beam using
         Fraunhofer diffraction.
        :param samples: matrix of the samples that will be transformed
        """
        self.samples = samples
        self.xdim = samples.shape[0]
        self.ydim = samples.shape[1]
        self.f = f
        self.z = z
        self.N = int(self.xdim/2)
        self.wavelength = c / self.f

        self.dx = 0.001
        self.dy = 0.001
        self.dx2 = 1
        self.dy2 = 1
        self.A = ( self.xdim*self.dx*self.dx2 )/ ( self.wavelength*self.z )
        self.amp = 1

    def __DFT_1D(self, x):
        """
        Appllies the Slow (normal) Fourier Trasnformation in 1D using numpy arrays.
        :param x: 1D array for which the DFT is to be evaluated
        :return: value of the DFT in the specified points
        """
        N = x.shape[0]
        M = int(self.amp*N/2)
        n = np.arange(-M, M, 1)
        k = n.reshape((N, 1)) # Vertical vector to be able to create the M(m x n) array. We defined k from -M to M to get a zero-centered image
        M = np.exp( -2j*np.pi*self.A*k*n / N )

        return np.dot(M, x)

    def FFT_1D(self, x):
        """
        Appllies the Fast Fourier Trasnformation in 1D. Implementation of
         the Cooley-Turkey algorithm.
        :param x: 1D array for which the DFT is to be evaluated
        :return: value of the DFT in the specified points
        """
        x = np.asarray(x, dtype=np.complex64)
        N = x.shape[0]
        M = int(self.amp*N/2)
        
        if N % 2 > 0:
            raise ValueError("X dimension must be a power of 2")
        elif N <= N: # The cutoff should be optimized (depending on the value it could produce 2N images of the results)
            return self.__DFT_1D(x)
        else:
            x_even = self.FFT_1D(x[::2]) # We take the even indices
            x_odd = self.FFT_1D(x[1::2]) # Odd indices
            factor = np.exp( -2j*np.pi*self.A*np.arange(N) / (N) ) # REVISAR SI ESTÁN BIEN LOS FACTORES DE /2
            return np.concatenate([x_even + factor[:M]*x_odd, x_even + factor[M:]*x_odd]) # ANTES SE TOMABA factor[:M//2]
        
    def FFT_2D(self, xy):
        """
        Appllies the Fast Fourier Transformation in 2D. Implementation of
         the Cooley-Turkey algorithm.
        :param xy: 2D array for which the DFT is to be evaluated
        :return: value of the DFT in the specified points
        """
        fft = np.empty((self.amp*xy.shape[0], self.amp*xy.shape[1]), dtype=np.complex64)
        y = np.empty((xy.shape[0], self.amp*xy.shape[1]), dtype=np.complex64)
        xy = np.asarray(xy, dtype=np.complex64)

        for i in range(xy.shape[0]):
            y[i,:] = self.FFT_1D(xy[i,:])
        for j in range(self.amp*xy.shape[1]):
            fft[:,j] = self.FFT_1D(y[:,j])
        
        return fft
       
    def diffraction(self):
        """
        Propagates the initial beam using the Fraunhofer diffraction equation
        :return: array of xdim x ydim dimensions for the propagted beam
        """
        k = 2*np.pi/self.wavelength
        A = np.exp( 1j*k*self.z )/( 1j*self.wavelength*self.z )

        return A*self.FFT_2D(self.samples)*self.dx2*self.dy2
        return A*np.fft.fftshift(np.fft.fft2(self.samples))*self.dx2*self.dy2