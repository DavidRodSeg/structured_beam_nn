# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scalar_monobeam import GaussianBeam, HGBeam, LGBeam, BGBeam
import time
from mask import Mask
from fraunhofer_diffraction import Fraunhofer
import warnings
warnings.filterwarnings("ignore")



# --------------- REPRESENTATION OF THE INTENSITY PROFILE ---------------
beam = LGBeam(10, f=600*10e6, E0=10, p=0, l=0)
print("Creating field/intensity matrices... ")
t0 = time.time()
MatE = beam.Propagate(z2=10.05, dz=10,r2=64, dr = 1, select=True)
print(f"Done. Time needed: {time.time()-t0} seconds")

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



#--------------------------Apply the mask-----------------------------
t0 = time.time()
print("Applying the mask...")
MatE = np.squeeze(MatE)
mask = Mask(MatE.shape[0], MatE.shape[1]).setCircular(10)
product = mask.Apply(MatE)
MatI = beam.Module(product)
print(f"Done. Time needed: {time.time()-t0} seconds")

plt.imshow(MatI, cmap="grey").set_clim(vmin=0, vmax=1e-5)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.colorbar()
plt.show()



#----------------------Fraunhofer diffraction-------------------------
t0 = time.time()
print("Propagating the beam...")
diff = Fraunhofer(samples=product, f=600*10e6, z = 1)
diff_arr = diff.diffraction()
MatI = beam.Module(diff_arr)
print(f"Done. Time needed: {time.time()-t0} seconds")

plt.imshow(MatI, cmap="grey").set_clim(vmin=0, vmax=1e-4)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.colorbar()
plt.show()