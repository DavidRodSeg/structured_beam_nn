import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_intensity(image):
    return np.sum(image)

def find_background(angles):
    return np.argmin(angles)


df = pd.read_csv("slm_preprocessed/slm_preprocessed.csv", header=None, sep=";")
df2 = pd.read_csv("sin_slm_preprocessed/sin_slm_preprocessed.csv", header=None, sep=";")
df3 = pd.read_csv("laser_profile_preprocessed/laser_profile_preprocessed.csv", header=None, sep=";")



images_slm = df.values[:,:-1].reshape(-1, 256, 256)
angles_slm = df.values[:,-1]

images_sin_slm = df2.values[:,:-1].reshape(-1, 256, 256)
angles_sin_slm = df2.values[:,-1]

images_laser = df3.values[:,:-1].reshape(-1, 256, 256)
angles_laser = df3.values[:,-1]

evolution = []
evolution2 = []
evolution3 = []

for image in images_slm:
    evolution.append(calculate_intensity(image[:-1]))

for image in images_sin_slm:
    evolution2.append(calculate_intensity(image[:-1]))

for image in images_laser:
    evolution3.append(calculate_intensity(image[:-1]))


evolution = np.array(evolution)
evolution2 = np.array(evolution2)
evolution3 = np.array(evolution3)

max_value = evolution3[find_background(angles_laser)]

evolution_normalized = evolution / np.max(evolution)
evolution2_normalized = evolution2 / np.max(evolution2)
evolution3_normalized = evolution3 / max_value

evolution3_normalized = np.delete(evolution3_normalized, find_background(angles_laser))
angles_laser = np.delete(angles_laser, find_background(angles_laser))


plt.scatter(angles_slm, evolution_normalized, color='blue', marker='o', label = "With SLM")
plt.scatter(angles_sin_slm, evolution2_normalized, color='red', marker='o', label = "Without SLM")
plt.ylim(0, 1)
plt.xlabel('Degrees of the analyzer (º)')
plt.ylabel('Intensity (normalized)')
plt.title('Intensity evolution')
plt.grid(True)
plt.legend()
plt.show()

plt.scatter(angles_laser, evolution3_normalized, color='blue', marker='o')
plt.ylim(0, 1)
plt.xlabel('Degrees of the analyzer (º)')
plt.ylabel('Intensity (normalized)')
plt.title('Intensity evolution')
plt.grid(True)
plt.legend()
plt.show()