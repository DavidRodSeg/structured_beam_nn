# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scalar_monobeam import GaussianBeam, HGBeam, LGBeam, BGBeam
import time
from mask import Mask
from fraunhofer_diffraction import Fraunhofer
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


times = 500 # Number of iterations for the creation of the dataset
mask_size = 256

dataset_2 = np.empty((times, mask_size**2+mask_size**2), dtype=np.float32)
dataset = np.empty((times, mask_size**2+3), dtype=np.float32)

for t in tqdm(range(times)):
    #-----------------------CREATION OF THE BEAM--------------------------
    # w_rand = np.random.uniform(0.0090, 0.0110)
    # if t < 500:
    #     randp = np.random.randint(0,5)
    #     randl = np.random.randint(0,5)
    #     ind = 0
    #     beam = LGBeam(0.01, f=5.63*10e14, E0=1, p=randp, l=randl) # 0.01, 600, 0.5. Realmente la frecuencia debería ser 5.63*10e14 porque lambda = 532*10e-9
    # elif t >= 500 and t < 1000:
    #     randp = np.random.randint(0,5)
    #     randl = np.random.randint(0,5)
    #     ind = 1
    #     beam = HGBeam(0.01, f=5.63*10e14, E0=1, p=randp, m=randl)
    # else:
    #     randp = np.random.randint(0,10)
    #     randl = 0
    #     ind = 2
    #     beam = BGBeam(0.01, f=5.63*10e14, E0=100, beta=randp)
    beam = LGBeam(0.03, f=5.63*1e14, E0=100, p=0, l=0)
    t0 = time.time()
    MatE = beam.Propagate(z1=0, z2=0, dz=0.01, r2=mask_size/2*35.5*1e-6, dr = 35.5*1e-6, select=True) # 0.09, 0.1, 0.01, 0.128, 0.001
    
    # MatI_field = beam.Module(MatE.reshape(mask_size, mask_size))
    # MatPhase_field = beam.phase_array(MatE.reshape(mask_size, mask_size))

    # plt.imshow(MatI_field, cmap="grey")#.set_clim(vmin=0, vmax=1e-5)
    # plt.xlabel('X-axis px')
    # plt.ylabel('Y-axis px')
    # plt.colorbar()
    # plt.show()

    # plt.imshow(MatPhase_field, cmap="grey")#.set_clim(vmin=0, vmax=1e-5)
    # plt.xlabel('X-axis px')
    # plt.ylabel('Y-axis px')
    # plt.colorbar()
    # plt.show()

    #--------------------------APPLY THE MASK-----------------------------
    MatE = np.squeeze(MatE)

    # randSide = np.random.randint(3,8)
    # randAngle = np.random.uniform(0, 2*np.pi)
    randN1 = np.random.randint(10, 40)
    # randN2 = np.random.randint(10, 100)
    # rand1 = np.random.uniform(-15, 15)
    # rand2 = np.random.uniform(-15, 15)
    # mask = Mask(MatE.shape[0], MatE.shape[1]).setRegularPolygon(randN1, randSide, center=(rand1,rand2)).rotation(randAngle)#.noise()
    mask = Mask(MatE.shape[0], MatE.shape[1]).setSlit(N=randN1, size=5).noise()
    mask_arr = mask.get() # Necessary for the creation of the dataset
    mask_arr_int = beam.Module(mask_arr)
    product = mask.Apply(MatE)
    MatI = beam.Module(product)

    # plt.imshow(mask_arr, cmap="grey")#.set_clim(vmin=0, vmax=1e-5)
    # plt.xlabel('X-axis label')
    # plt.ylabel('Y-axis label')
    # plt.colorbar()
    # plt.show()



    #----------------------FRAUNHOFER DIFFRACTION-------------------------
    # z_rand = np.random.uniform(0.195, 0.205)
    diff = Fraunhofer(samples=product, f=5.63*1e14, z = 0.2) # 10000
    diff_arr = diff.diffraction(padding=True)
    # diff_arr = fraunhofer(product)
    MatI = beam.Module(diff_arr)
    maximum = np.max(MatI)
    MatI = MatI/maximum

    # plt.imshow(MatI, cmap="grey").set_clim(vmin=0, vmax=1e-3)
    # plt.xlabel('X-axis px')
    # plt.ylabel('Y-axis px')
    # plt.colorbar()
    # plt.show()

    # vector = range(mask_size)
    # plt.figure()
    # plt.plot(MatI[vector, vector])
    # plt.xlabel('(y = x)-axis px')
    # plt.ylabel('Intensity (normalized)')
    # plt.show()

    # plt.figure()
    # plt.plot(MatI[:, int(mask_size/2)])
    # plt.xlabel('X-axis px')
    # plt.ylabel('Intensity (normalized)')
    # plt.show()

    #-----------------------CREATING THE DATASET--------------------------
    dataset_2[t, :] = np.concatenate((MatI.flatten("C"), mask_arr.flatten("C")))

    # t_data = np.hstack((MatI.flatten(), [randN1, 0, 1])).reshape(1,-1)
    # if t == 0:
    #     dataset = t_data
    # else:
    #     dataset = np.concatenate((dataset, t_data), axis=0)


#-----------------------SAVING THE DATASET------------------------
# df = pd.DataFrame(dataset)
df2 = pd.DataFrame(dataset_2)
# df.to_csv(f"circular_parameters.csv", sep=";", header=False, index=False)
df2.to_csv(f"slit_images.csv", sep=";", header=False, index=False)
# df.to_csv("modeltesting_circular_parameters.csv", sep=";", header=False, index=False)
# df.to_csv(f"classification.csv", sep=";", header=False, index=False)
