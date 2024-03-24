"""
Classes and subclasses for the calculation of the amplitude and phase of different structured light beam. The parameters are stored in
 arrays of a specified size. For more information check out the description of each of the classes and its attributes.
"""



# Import libraries
import numpy as np
from scipy.special import hermite, genlaguerre, jv



# Define the gaussian beam class and derived classes
DEFAULT_E0 = 1
DEFAULT_PHI0 = 0
DEFAULT_ALPHA = 0
c = 3*10**8 # All units are in the International System
μ0 = 4*np.pi*10**(-7)

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
        self.I0_norm = 1/( 2*μ0*c )
        self.FWHM = self.w0 * np.sqrt( 2*np.log(2) )
        self.z0 = np.pi*self.w0**2 / self.wave_length
        self.k = 2*np.pi/self.wave_length
        self.M = 1 + 0j

    def getWaist(self, z):
        """
        Calculate the beam waist width for an specific z value.
        :param z: coordinate z  in cylindrical coordinates
        :return: Beam's waist width w(z)
        """
        w = self.w0 * np.sqrt( 1 + ( self.wave_length*z / ( np.pi*self.w0**2 )**2 ))

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
            R = z + ( np.pi*self.w0**2 / self.wave_length )**2 / z
        
        return R
    
    def getGouyPhase(self, z):
        """
        Calculates the Gouy phase of the beam.
        :param z: coordinate z in cylindrical coordinates
        :return: Beam's Gouy's phase gphi(z) in radians
        """
        gphi = np.arctan( z / self.z0)

        return gphi
    
    def getFieldAmplitude(self, z, r):
        """
        Calculates the amplitude of the beam's field.
        :param z: coordinate z in cylindrical coordinates
        :param r: coordinate r in cylindrical coordinates
        :return: Beam's amplitude A(r,z)
        """
        w = self.getWaist(z)
        A = self.E0 * ( self.w0 / w ) * np.exp( -r**2 / w**2 )

        return A

    def getFieldPhase(self, z, r):
        """
        Calculates the phase of the beam's field.
        :param z: coordinate z in cylindrical coordinates
        :param r: coordinate r in cylindrical coordinates
        :return: Beam's phase phi(r,z)
        """
        R = self.getRadius(z)
        gphi = self.getGouyPhase(z)
        phi = -self.k*z - np.pi*r**2 / ( self.wave_length*R ) - gphi

        return phi

    def getFieldVector(self, z, r): # REVISAR SI ESTÁ BIEN LA SALIDA EN CUANTO A SISTEMA DE COORDENADAS
        """
        Gives the field vector (in the cross-section plane) at the position
         (r, z) in cylindrical coordinates. 
        :param z: coordinate z in cylindrical coordinates
        :param r: coordinate r in cylindrical coordinates
        :return: Beam's field vector E(r,z). The polarization is represented in cartesian coordinates, as the function
         returns a vector of two components: one for the X axis and another one for the Y axis.
        """
        A = self.getFieldAmplitude(z, r)
        phi = self.getFieldPhase(z, r)
        A_complex = A * np.exp( 1j*phi )
        if self.alpha==0:
            E = A_complex
        else:
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
        I = self.I0_norm * abs(A)**2

        return I

    def Propagate(self, z1=0.05, z2=1.05, dz=1, r2=10, dr=0.1, select=False): # Raidus r in centimeters
        """
        Propagate the beam in free space. Calculates the fundamental parameters
         of the beam for a certain distance and store it in a vector.
        :param z1: coordinate z for the starting point of the propagation
        :param z2: coordinate z for the final point of the propagation
        :param dz: step distance of the propagation
        :param r2: final point for the coordinate r
        :param dr: step distance for the coordinate r
        :param select: if False calculates the intensity array. If True calculates the field array
        :return: matrices of the intensity and the field vector of the beam
         MatE, MatI with dimensions (z_steps, 2*r_steps, 2*r_steps, 2) and
         (z_steps, 2*r_steps, 2*r_steps) respectively
        """
        # Number of steps for the for loops
        z_steps = int(( z2-z1 ) / dz)
        r_steps = int( r2 / dr)
        c_steps = 2*r_steps #Cartesian steps: twice the steps of the radius

        # Intensity and field vector matrices for the specified range
        if select == False:
            Mat = np.empty((z_steps, c_steps, c_steps), dtype=np.float32)

            for i in range(z_steps):
                for j in range(c_steps):
                    for k in range(c_steps):
                        z = i * dz + z1
                        x = j * dr - r2
                        y = k * dr - r2
                        r = np.sqrt(x**2 + y**2)
                        Iijk = self.getIntensity(z, r)
                        Mat[i, j, k] = Iijk

        elif select == True:
            if self.alpha==0:
                Mat = np.empty((z_steps, c_steps, c_steps), dtype=np.complex64)
            else:
                Mat = np.empty((z_steps, c_steps, c_steps, 2), dtype=np.complex64)

            for i in range(z_steps):
                for j in range(c_steps):
                    for k in range(c_steps):
                        z = i * dz + z1
                        x = j * dr - r2
                        y = k * dr - r2
                        r = np.sqrt(x**2 + y**2)
                        Eijk = self.getFieldVector(z,r)
                        Mat[i, j, k] = Eijk

        else:
            print("Error: Not a valid selection")
  
        return Mat
    
    def Rotate(self, alpha):
        """
        Rotates the polarization of the beam.
        :param alpha: rotation angle (°)
        :return: GaussianBeam object for chaining
        """
        self.alpha = alpha

        return self

    def __Mask(self, M=None):
        """
        Apply a complex mask to modulate the beam amplitude and phase.
        :param M: mask function that modulate the beam
        :return: GaussianBeam object for chaining
        """
        M = self.M if M==None else M

        return M
    
    def Module(self, array):
        """
        Transform the field's array in an array of the intensity vales.
        :param array: field's array
        :return: array of intensities
        """
        intensity = np.empty((array.shape[0], array.shape[1]), dtype=np.float32)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                intensity[i,j] = self.I0_norm * abs( array[i,j] )**2
        
        return intensity



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
    
    def getFieldPhase(self, x, y, z):
        """
        Calculates the phase of the beam's field.
        :param x: coordinate x in cartesian coordinates
        :param y: coordinate y in cartesian coordinates
        :param z: coordinate z in cartesian coordinates
        :return: Beam's phase phi(x, y, z)
        """
        R = self.getRadius(z)
        gphi = self.getGouyPhase(z)
        phi = -self.k*z - self.k*( x**2+y**2 ) / ( 2*R ) - gphi

        return phi
    
    def getFieldVector(self, x, y, z):
        """
        Gives the field vector (in the cross-section plane) at the position
         (x, y, z) in cartesian coordinates. 
        :param x: coordinate x in cartesian coordinates
        :param y: coordinate y in cartesian coordinates
        :param z: coordinate z in cartesian coordinates
        :return: Beam's field vector E(z, y, z)
        """
        A = self.getFieldAmplitude(x, y, z)
        phi = self.getFieldPhase(x, y, z)
        A_complex = A * np.exp( 1j*phi )
        if self.alpha==0:
            E = A_complex
        else:
            E = np.array([A_complex*np.cos( self.alpha ), A_complex*np.sin( self.alpha )])

        return E
    
    def getIntensity(self, x, y, z):
        """
        Calculate the intensity of the beam at the position (r, z).
        :param x: coordinate x in cartesian coordinates
        :param y: coordinate y in cartesian coordinates
        :param z: coordinate z in cartesian coordinates
        :return: Beam's intensity I(x, y, z)
        """
        A = self.getFieldAmplitude(x, y, z)
        I = self.I0_norm * abs(A)**2

        return I
    
    def Propagate(self, z1=0.05, z2=1.05, dz=1, r2=10, dr=0.1, select=False):
        """
        Propagate the beam in free space. Calculates the fundamental parameters
         of the beam for a certain distance and store it in a vector.
        :param z1: coordinate z for the starting point of the propagation
        :param z2: coordinate z for the final point of the propagation
        :param dz: step distance of the propagation
        :param r2: final point for the coordinate r
        :param dr: step distance for the coordinate r
        :param select: if False calculates the intensity array. If True calculates the field array
        :return: matrices of the intensity and the field vector of the beam
         MatE, MatI with dimensions (z_steps, 2*r_steps, 2*r_steps, 2) and
         (z_steps, 2*r_steps, 2*r_steps) respectively
        """
        # Number of steps for the for loops
        z_steps = int(( z2-z1 ) / dz)
        r_steps = int( r2 / dr)
        c_steps = 2*r_steps #Cartesian steps: twice the steps of the radius

        # Intensity and field vector matrices for the specified range
        if select == False:
            Mat = np.empty((z_steps, c_steps, c_steps), dtype=np.float32)

            for i in range(z_steps):
                for j in range(c_steps):
                    for k in range(c_steps):
                        z = i * dz + z1
                        x = j * dr - r2
                        y = k * dr - r2
                        r = np.sqrt(x**2 + y**2)
                        Iijk = self.getIntensity(x, y, z)
                        Mat[i, j, k] = Iijk

        elif select == True:
            if self.alpha==0:
                Mat = np.empty((z_steps, c_steps, c_steps), dtype=np.complex64)
            else:
                Mat = np.empty((z_steps, c_steps, c_steps, 2), dtype=np.complex64)

            for i in range(z_steps):
                for j in range(c_steps):
                    for k in range(c_steps):
                        z = i * dz + z1
                        x = j * dr - r2
                        y = k * dr - r2
                        r = np.sqrt(x**2 + y**2)
                        Eijk = self.getFieldVector(x, y, z)
                        Mat[i, j, k] = Eijk

        else:
            print("Error: Not a valid selection")
  
        return Mat
    


DEFAULT_L = 0
DEFAULT_P = 0

class LGBeam(GaussianBeam):
    """
    Define a coherent monocromatic beam with a laguerre-gaussian cross-sectional profile
     and a linear polarization.
    :param W0: beam waist radius at the origin z=0 (mm)
    :param f: frequency of the monocromatic beam (THz)
    :param E0: intensity at the origin (N/C)
    :param phi0: initial phase
    :param alpha: the angle the polarization forms with the X-axis
    :param p: order of the LG beam (p = 0 by defect)
    :param l: order of the LG beam (l = 0 by defect). It is also the OAM (orbital angular
     momentum) of the vortex beam
    """
    def __init__(self, w0, f, E0=DEFAULT_E0, phi0=DEFAULT_PHI0, alpha=DEFAULT_ALPHA, p=DEFAULT_P, l=DEFAULT_L):
        super().__init__(w0, f, E0, phi0, alpha)
        self.l = l
        self.p = p 
    
    def getGouyPhase(self, z):
        gphi = super().getGouyPhase(z) * ( 1 + self.l + 2*self.p)

        return gphi
    
    def getFieldAmplitude(self, z, r):
        w = self.getWaist(z)
        Lp = genlaguerre(self.p, self.l)
        A = super().getFieldAmplitude(z, r) * ( np.sqrt(2)* r/w )**self.l * Lp( 2*r**2/w**2 )

        return A
    
    def getFieldPhase(self, z, r, theta):
        """
        Calculates the phase of the beam's field.
        :param z: coordinate z in cylindrical coordinates
        :param r: coordinate r in cylindrical coordinates
        :param theta: coordinate theta in cylindrical coordinates
        :return: Beam's phase phi(r,z, theta)
        """
        phi = super().getFieldPhase(z, r) + self.l*theta

        return phi
    
    def getFieldVector(self, z, r, theta):
        """
        Gives the field vector (in the cross-section plane) at the position
         (r, z) in cylindrical coordinates. 
        :param z: coordinate z in cylindrical coordinates
        :param r: coordinate r in cylindrical coordinates
        :param theta: coordinate theta in cylindrical coordinates
        :return: Beam's field vector E(r,z)
        """
        A = self.getFieldAmplitude(z, r)
        phi = self.getFieldPhase(z, r, theta)
        A_complex = A * np.exp( 1j*phi )
        if self.alpha==0:
            E = A_complex
        else:
            E = np.array([A_complex*np.cos( self.alpha ), A_complex*np.sin( self.alpha )])

        return E
    
    def Propagate(self, z1=0.05, z2=1.05, dz=1, r2=10, dr=0.1, select=False):
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
        r_steps = int( r2 / dr)
        c_steps = 2*r_steps #Cartesian steps: twice the steps of the radius

        # Intensity and field vector matrices for the specified range
        if select == False:
            Mat = np.empty((z_steps, c_steps, c_steps), dtype=np.float32)

            for i in range(z_steps):
                for j in range(c_steps):
                    for k in range(c_steps):
                        z = i * dz + z1
                        x = j * dr - r2
                        y = k * dr - r2
                        r = np.sqrt(x**2 + y**2)
                        Iijk = self.getIntensity(z, r)
                        Mat[i, j, k] = Iijk

        elif select == True:
            if self.alpha==0:
                Mat = np.empty((z_steps, c_steps, c_steps), dtype=np.complex64)
            else:
                Mat = np.empty((z_steps, c_steps, c_steps, 2), dtype=np.complex64)

            for i in range(z_steps):
                for j in range(c_steps):
                    for k in range(c_steps):
                        z = i * dz + z1
                        x = j * dr - r2
                        y = k * dr - r2
                        r = np.sqrt(x**2 + y**2)
                        if r == 0:
                            theta = 0 # Impose 0 to avoid problems in the IND 0/0 at the origin (r=0). This sentence has no relevance as the amplitude is 0 at r=0.
                        else:
                            theta = np.arcsin( y/r )

                        Eijk = self.getFieldVector(z, r, theta)
                        Mat[i, j, k] = Eijk

        else:
            print("Error: Not a valid selection")
  
        return Mat
    


DEFAULT_beta = 0
DEFAULT_ORDER = 0

class BGBeam(GaussianBeam):
    """
    Define a coherent monocromatic beam with a bessel-gaussian cross-sectional profile
     and a linear polarization.
    :param W0: beam waist radius at the origin z=0 (mm)
    :param f: frequency of the monocromatic beam (THz)
    :param E0: intensity at the origin (N/C)
    :param phi0: initial phase
    :param alpha: the angle the polarization forms with the X-axis
    :param beta: constant scale parameter
    :param order: order of the bessel function multiplying the field equation
    """
    def __init__(self, w0, f, E0=DEFAULT_E0, phi0=DEFAULT_PHI0, alpha=DEFAULT_ALPHA, beta=DEFAULT_beta, order=DEFAULT_ORDER):
        super().__init__(w0, f, E0, phi0, alpha)
        self.beta = beta
        self.order = order
    
    def getFieldAmplitude(self, z, r):
        A = super().getFieldAmplitude(z, r) * abs( jv(self.order, self.beta*r/( 1+1j*z/self.z0 )) ) * np.exp( -self.beta**2*(z/2*self.k) / ( 1+z/self.z0 ))

        return A
    
    def getFieldPhase(self, z, r):
        """
        Calculates the phase of the beam's field.
        :param z: coordinate z in cylindrical coordinates
        :param r: coordinate r in cylindrical coordinates
        :return: Beam's phase phi(r,z)
        """
        bessel_phase = np.arctan( np.real( jv(self.order, self.beta*r / ( 1+1j*z/self.z0 ) )) / np.imag( jv(self.order, self.beta*r / ( 1+1j*z/self.z0 ) )))
        phi = super().getFieldPhase(z, r) + bessel_phase + self.beta*( z**2/( 2*self.k*self.z0 ))/( 1+z/self.z0 )

        return phi