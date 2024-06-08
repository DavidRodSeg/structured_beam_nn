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
diff_size = 4*mask_size

# dataset = np.empty((times, diff_size+mask_size), dtype=np.float32)
dataset = np.empty((times, mask_size**2+2), dtype=np.float32)

for t in tqdm(range(times)):
    #-----------------------CREATION OF THE BEAM--------------------------
    randNp = np.random.randint(0, 5)
    randNl = np.random.randint(0, 5)
    beam = LGBeam(0.01, f=600*10e9, E0=0.5, p=randNp, l=randNl)
    # print("Creating field/intensity matrices... ")
    t0 = time.time()
    MatE = beam.Propagate(z1=0.09, z2=0.1, dz=0.01, r2=0.128, dr = 0.001, select=True)
    # print(f"Done. Time needed: {time.time()-t0} seconds")

    # Create the figure and axes
    # print("Starting the plotting...")
    # figure, axes = plt.subplots()
    # data = np.squeeze(MatI[0, :, :])
    # img = axes.imshow(data, cmap="gray")
    #img.set_clim(vmin=0, vmax=1e5)

    # Define the slider parameters
    # ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])  # Adjust the slider position and size as needed
    # slider = Slider(ax_slider, 'Index', 0, MatI.shape[0] - 1, valinit=0, valstep=1)

    # Define the update function for the slider
    # def update(val):
    #     index = int(val)
    #     img.set_data(MatI[index, :, :])
    #     figure.canvas.draw_idle()

    # Connect the update function to the slider
    # slider.on_changed(update)

    # Add labels to axes
    # axes.set_xlabel('X-axis label')
    # axes.set_ylabel('Y-axis label')

    # Display the plot
    # plt.colorbar(img)
    # xaxis = np.arange(-0.128, 0.128, 0.001)
    # plt.figure()
    # plt.plot(xaxis, MatI[0,0,:])
    # plt.title("Sección transversal")
    # plt.ylabel("Intensidad (J/(m*s))")
    # plt.xlabel("x (m)")
    # plt.show()



    #--------------------------APPLY THE MASK-----------------------------
    contador = np.zeros((250,), dtype=np.int32)
    # print("Applying the mask...")
    MatE = np.squeeze(MatE)
    if t < 500:
    #     randN1 = np.random.randint(10, MatE.shape[0])
    #     randN2 = np.random.randint(10, MatE.shape[1])
    #     mask = Mask(MatE.shape[0], MatE.shape[1]).setRectangle(a=randN1, b=randN2)
    # elif t >= 100 and t < 200:
        # randN1 = np.random.randint(0, 128)
        # mask = Mask(MatE.shape[0], MatE.shape[1]).setCircular(r=randN1)
    # elif t >= 200 and t < 300:
        randN1 = np.random.randint(0, 128)
        mask = Mask(MatE.shape[0], MatE.shape[1]).setSlit(N=randN1)

    # mask = Mask(MatE.shape[0], MatE.shape[1]).setRectangle(a=50, b = 50)
    mask_arr = mask.get() # Necessary for the creation of the dataset
    mask_arr_int = beam.Module(mask_arr)
    product = mask.Apply(MatE)
    MatI = beam.Module(product)
    # print(f"Done. Time needed: {time.time()-t0} seconds")

    # plt.imshow(MatI, cmap="grey")#.set_clim(vmin=0, vmax=1e-5)
    # plt.xlabel('X-axis label')
    # plt.ylabel('Y-axis label')
    # plt.colorbar()
    # plt.show()



    #----------------------FRAUNHOFER DIFFRACTION-------------------------
    # print("Propagating the beam...")
    # t0 = time.time()
    diff = Fraunhofer(samples=product, f=600*10e9, z = 10000) # z = 10
    diff_arr = diff.diffraction()
    MatI = beam.Module(diff_arr)
    # print(f"Time: {time.time()-t0} seconds")
    # print(f"Done. Time needed: {time.time()-t0} seconds")

    # plt.imshow(MatI, cmap="grey")#.set_clim(vmin=0, vmax=1e-3)
    # plt.xlabel('X-axis label')
    # plt.ylabel('Y-axis label')
    # plt.colorbar()
    # plt.show()

    # plt.figure()
    # plt.plot(MatI[int(mask_size/2), :])
    # plt.xlabel('X-axis label')
    # plt.ylabel('Y-axis label')
    # plt.show()

    # plt.figure()
    # plt.plot(MatI[:, int(mask_size/2)])
    # plt.xlabel('X-axis label')
    # plt.ylabel('Y-axis label')
    # plt.show()


    #-----------------------CREATING THE DATASET--------------------------
    # dataset[t, :] = np.concatenate((MatI.flatten("C"), mask_arr_int.flatten("C")))
    # t_data = np.concatenate((MatI, mask_arr), axis=1)
    # if t == 0:
    #     dataset = t_data
    # else:
    #     dataset = np.concatenate((dataset, t_data), axis=0)

    t_data = np.hstack((MatI.flatten(), [randN1, randNp, randNl])).reshape(1,-1)
    if t == 0:
        dataset = t_data
    else:
        dataset = np.concatenate((dataset, t_data), axis=0)

    # if t < 100:
    #     t_data = np.hstack((MatI.flatten(), 1)).reshape(1,-1)
    #     if t == 0:
    #         dataset = t_data
    #     else:
    #         dataset = np.concatenate((dataset, t_data), axis=0)
    # elif t >= 100 and t < 200:
    #     t_data = np.hstack((MatI.flatten(), 2)).reshape(1,-1)
    #     if t == 0:
    #         dataset = t_data
    #     else:
    #         dataset = np.concatenate((dataset, t_data), axis=0)
    # else:
    #     t_data = np.hstack((MatI.flatten(), 3)).reshape(1,-1)
    #     if t == 0:
    #         dataset = t_data
    #     else:
    #         dataset = np.concatenate((dataset, t_data), axis=0)
        
#-----------------------SAVING THE DATASET------------------------
df = pd.DataFrame(dataset)
df.to_csv(f"{times}_diffraction_intensity_slits_pl.csv", sep=";", header=False, index=False)
# df.to_csv("modeltesting_intensity_rectangle_2.csv", sep=";", header=False, index=False)
# df.to_csv(f"classification.csv", sep=";", header=False, index=False)
