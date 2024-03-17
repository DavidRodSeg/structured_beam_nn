# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite

# Define the gaussian beam
DEFAULT_E0 = 1
DEFAULT_PHI0 = 0
DEFAULT_ALPHA = 0
c = 3*10**8 # All units are in the International System
pi = 4*np.arctan(1)
μ0 = 4*pi*10**(-7)

class GaussianBeam:
    def __init__(self, w0, f, E0=DEFAULT_E0, phi0=DEFAULT_PHI0, alpha=DEFAULT_ALPHA):
        """"
        Define a coherent monocromatic beam with a gaussian cross-sectional profile
         and a linear polarization.
        :param W0: beam waist radius at the origin z=0 (mm)
        :param f: frequency of the monocromatic beam (THz)
        :param E0: intensity at the origin (N/C)
        :param phi0: initial phase
        :param alpha: the angle the polarization forms with the X-axis
        """
        self.w0 = w0
        self.f = f
        self.E0 = E0
        self.phi0 = phi0
        self.alpha = alpha

        self.wave_length = c / self.f
        self.I0 = 1/( 2*μ0*c ) * abs(self.E0)**2 # PENSANDO EN QUITARLO
        self.FWHM = self.w0 * np.sqrt( 2*np.log(2) )
        self.M = 1 + 0j

        # self.z = 0
        # self.z0 = 0
        # self.r = 0

    def getWaist(self, z):
        """
        Calculate the beam waist width for an specific z value.
        :param z: coordinate z  in cylindrical coordinates
        :return: Beam's waist width w(z)
        """
        w = self.w0 * np.sqrt( 1 + ( self.wave_length*z / ( pi*self.w0**2 )**2 ))

        return w
    
    def getRadius(self, z):
        """
        Calculates the curvature radius of the beam.
        :param z: coordinate z in cylindrical coordinates
        :return: Beam's curvature radius R(z)
        """
        w = self.getWaist(z)
        
        if z == 0:
            R = float("inf")
        else:
            R = z + ( pi*self.w0**2 / self.wave_length )**2 / z
        
        return R
    
    def getGouyPhase(self, z):
        """
        Calculates the Gouy phase of the beam.
        :param z: coordinate z in cylindrical coordinates
        :return: Beam's Gouy's phase gphi(z) in radians
        """
        gphi = np.arctan( self.wave_length*z / ( pi*self.w0**2 ))

        return gphi
    
    def getFieldAmplitude(self, z, r):
        """
        Calculates the amplitude of the beam's field.
        :param z: coordinate z in cylindrical coordinates
        :param r: coordinate r in cylindrical coordinates
        :return: Beam's amplitude A(r,z)
        """
        w = self.getWaist(z)
        Amp_mask = abs(self.M)
        A = self.E0 * ( self.w0 / w ) * np.exp( -r**2 / w**2 ) * Amp_mask

        return A

    def getFieldPhase(self, z, r): # Puede que dé problemas
        """
        Calculates the phase of the beam's field.
        :param z: coordinate z in cylindrical coordinates
        :param r: coordinate r in cylindrical coordinates
        :return: Beam's phase phi(r,z)
        """
        k = 2*pi / self.wave_length
        w = self.getWaist(z)
        R = self.getRadius(z)
        gphi = self.getGouyPhase(z)
        mask_phase = np.arctan( np.imag(self.M) / np.real(self.M))
        phi = -k*z - pi*r**2 / ( self.wave_length*R ) + gphi + mask_phase

        return phi

    def getFieldVector(self, z, r): # REVISAR SI ESTÁ BIEN LA SALIDA EN CUANTO A SISTEMA DE COORDENADAS
        """
        Gives the field vector (in the cross-section plane) at the position
         (r, z) in cylindrical coordinates. 
        :param z: coordinate z in cylindrical coordinates
        :param r: coordinate r in cylindrical coordinates
        :return: Beam's field vector E(r,z)
        """
        A = self.getFieldAmplitude(z, r)
        phi = self.getFieldPhase(z, r)
        A_complex = A * np.exp( 1j*phi )
        E = np.array([A_complex*np.cos( self.alpha ), A_complex*np.sin( self.alpha )])

        return E

    def getIntensity(self, z, r):
        """
        Calculate the intensity of the beam at the position (r, z).
        :param z: coordinate z in cylindrical coordinates
        :param r: coordinate r in cylindrical coordinates
        :return: Beam's intensity I(r,z)
        """
        A = self.getFieldAmplitude(z, r)
        I = 1/( 2*μ0*c ) * abs(A)**2

        return I

    def Propagate(self, z1, z2, dz, r2=10, dr=0.1): # Raidus r in centimeters
        """
        Propagate the beam in free space. Calculates the fundamental parameters
         of the beam for a certain distance and store it in a vector.
        :param z1: coordinate z for the starting point of the propagation
        :param z2: coordinate z for the final point of the propagation
        :param dz: step distance of the propagation
        :param r2: final point for the coordinate r
        :param dr: step distance for the coordinate r
        :return: matrices of the intensity and the field vector of the beam
         MatE, MatI with dimensions (z_steps, 2*r_steps, 2*r_steps, 2) and
         (z_steps, 2*r_steps, 2*r_steps) respectively
        """
        # Number of steps for the for loops
        z_steps = int(( z2-z1 ) / dz)
        r_steps = int(( r2-0 ) / dr)

        # Initialization of the matrices
        MatE = np.empty((z_steps, 2*r_steps, 2*r_steps, 2), dtype=np.complex128)
        MatI = np.empty((z_steps, 2*r_steps, 2*r_steps), dtype=np.float32)

        # Intensity and field vector matrices for the specified range
        for i in range(z_steps):
            for j in range(2*r_steps):
                for k in range(2*r_steps):
                    z = i * dz
                    x = j * dr - r2
                    y = k * dr - r2
                    r = np.sqrt(x**2 + y**2)

                    Eijk = self.getFieldVector(z,r)
                    Iijk = self.getIntensity(z, r)

                    MatE[i, j, k] = Eijk
                    MatI[i, j, k] = Iijk
                
        return MatE, MatI
    
    def Rotate(self, alpha):
        """
        Rotates the polarization of the beam.
        :param alpha: rotation angle (°)
        :return: GaussianBeam object for chaining
        """
        self.alpha = alpha

        return self

    def Mask(self, M=None):
        """
        Apply a complex mask to modulate the beam amplitude and phase.
        :param M: mask function that modulate the beam
        :return: GaussianBeam object for chaining
        """
        M = self.M if M==None else M

        return M

    def plotIntensity(self, MatI):
        """
        Plot beam intensity profile.
        :param MatI: intensity profile matrix.
        :return: GaussianBeam object for chaining
        """
        self._plot("Intensity profile")
    
    def plotPhase(self):
        """
        Plot beam phase profile.
        """

    def plotAmplitude(self):
        """
        Plot beam amplitude profile
        """

    def plotField(self):
        """
        Plot beam vector field profile
        """


    @staticmethod

    def _plot(self, title, xlabel, ylabel, xvalue, yvalue):
        plt.plot(xvalue, yvalue)
        plt.title(title)
        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        plt.show()


DEFAULT_M = 0
DEFAULT_P = 0

class HGBeam(GaussianBeam):
    """
    Define a coherent monocromatic beam with a hermite-gaussian cross-sectional profile
         and a linear polarization.
        :param W0: beam waist radius at the origin z=0 (mm)
        :param f: frequency of the monocromatic beam (THz)
        :param E0: intensity at the origin (N/C)
        :param phi0: initial phase
        :param alpha: the angle the polarization forms with the X-axis
        :param m: order of the HG beam (m = 0 by defect)
        :param p: order of the HG beam (p = 0 by defect)
    """
    def __init__(self, w0, f, E0=DEFAULT_E0, phi0=DEFAULT_PHI0, alpha=DEFAULT_ALPHA, m=DEFAULT_M, p=DEFAULT_P):
        super().__init__(w0, f, E0, phi0, alpha)
        self.m = m
        self.p = p 
    
    def getGouyPhase(self, z):
        gphi = super().getGouyPhase(z) * ( 1 + self.m + self.p)

        return gphi
    
    def getFieldAmplitude(self, x, y, z):
        """
        Calculates the amplitude of the beam's field.
        :param x: coordinate x in cartesian coordinates
        :param y: coordinate y in cartesian coordinates
        :param z: coordinate z in cartesian coordinates
        :return: Beam's amplitude A(x, y, z)
        """
        hm = hermite(self.m)
        hp = hermite(self.p)
        w = self.getWaist(z)
        r = np.sqrt( x**2 + y**2 )

        A = hm( np.sqrt(2)*x / w ) * hp( np.sqrt(2)*y / w ) * super().getFieldAmplitude(z, r)

        return A