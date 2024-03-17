# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ScalarMonoBeam import GaussianBeam

# --------------- REPRESENTATION OF THE INTENSITY PROFILE ---------------
beam = GaussianBeam(1, 1000, 10)
MatE, MatI = beam.Propagate(0, 0.1, 0.001, r2=50, dr = 1)

# Create the figure and axes
figure, axes = plt.subplots()
data = MatI[0, :, :]
img = axes.imshow(data, cmap="gray")
img.set_clim(vmin=0, vmax=0.0001)

# Define the slider parameters
ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])  # Adjust the slider position and size as needed
slider = Slider(ax_slider, 'Index', 0, MatI.shape[0] - 1, valinit=0, valstep=1)

# Define the update function for the slider
def update(val):
    index = int(val)
    img.set_data(MatI[index, :, :])
    figure.canvas.draw_idle()

# Connect the update function to the slider
slider.on_changed(update)

# Add labels to axes
axes.set_xlabel('X-axis label')
axes.set_ylabel('Y-axis label')

# Display the plot
plt.colorbar(img)
plt.show()