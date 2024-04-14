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



times = 1 # Number of iterations for the creation of the dataset
mask_size = 128
diff_size = 4*mask_size

# dataset = np.empty((times, diff_size+mask_size), dtype=np.float32)
dataset = np.empty((times*mask_size, 2*mask_size), dtype=np.float32)

for t in tqdm(range(times)):
    #-----------------------CREATION OF THE BEAM--------------------------
    beam = LGBeam(0.01, f=600*10e9, E0=50, p=0, l=0)
    # print("Creating field/intensity matrices... ")
    t0 = time.time()
    MatE = beam.Propagate(z1=0.09, z2=0.1, dz=0.01, r2=0.064, dr = 0.001, select=True)
    # print(f"Done. Time needed: {time.time()-t0} seconds")

    # # Create the figure and axes
    # print("Starting the plotting...")
    # figure, axes = plt.subplots()
    # data = np.squeeze(MatI[0, :, :])
    # img = axes.imshow(data, cmap="gray")
    # img.set_clim(vmin=0, vmax=1e-5)

    # # Define the slider parameters
    # ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])  # Adjust the slider position and size as needed
    # slider = Slider(ax_slider, 'Index', 0, MatI.shape[0] - 1, valinit=0, valstep=1)

    # # Define the update function for the slider
    # def update(val):
    #     index = int(val)
    #     img.set_data(MatI[index, :, :])
    #     figure.canvas.draw_idle()

    # # Connect the update function to the slider
    # slider.on_changed(update)

    # # Add labels to axes
    # axes.set_xlabel('X-axis label')
    # axes.set_ylabel('Y-axis label')

    # # Display the plot
    # plt.colorbar(img)
    # plt.show()



    #--------------------------APPLY THE MASK-----------------------------
    # print("Applying the mask...")
    MatE = np.squeeze(MatE)
    randN = np.random.randint(0, 100)
    mask = Mask(MatE.shape[0], MatE.shape[1]).setSlit(N=randN)
    mask_arr = mask.get() # Necessary for the creation of the dataset
    mask_arr_int = beam.Module(mask_arr)
    product = mask.Apply(MatE)
    # MatI = beam.Module(product)
    # print(f"Done. Time needed: {time.time()-t0} seconds")

    # plt.imshow(MatI, cmap="grey")#.set_clim(vmin=0, vmax=1e-5)
    # plt.xlabel('X-axis label')
    # plt.ylabel('Y-axis label')
    # plt.colorbar()
    # plt.show()



    #----------------------FRAUNHOFER DIFFRACTION-------------------------
    # print("Propagating the beam...")
    diff = Fraunhofer(samples=product, f=600*10e9, z = 10)
    diff_arr = diff.diffraction()
    MatI = beam.Module(diff_arr)
    # print(f"Done. Time needed: {time.time()-t0} seconds")

    # plt.imshow(MatI, cmap="grey").set_clim(vmin=0, vmax=1e-4)
    # plt.xlabel('X-axis label')
    # plt.ylabel('Y-axis label')
    # plt.colorbar()
    # plt.show()


    #-----------------------CREATING THE DATASET--------------------------
    # dataset[t, :] = np.concatenate((MatI.flatten("C"), mask_arr_int.flatten("C")))
    t_data = np.concatenate((MatI, mask_arr_int), axis=1)
    if t == 0:
        dataset = t_data
    else:
        dataset = np.concatenate((dataset, t_data), axis=0)

#-----------------------SAVING THE DATASET------------------------
df = pd.DataFrame(dataset)
# df.to_csv(f"{times}_diffraction_intensity.csv", sep=";", header=False, index=False)
df.to_csv(f"modeltesting_intensity.csv", sep=";", header=False, index=False)