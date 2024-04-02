import numpy as np
import time

# Testing the maximum size of a complex numpy array
t0 = time.time()
n_rows = 500000
n_columns = 1000 # As it is a complex number it "has" twice the columns
array = np.random.uniform(-1, 1, size=(n_rows, n_columns)) + 1.j * np.random.uniform(-1, 1, size=(n_rows, n_columns))
# array = np.random.uniform(-1, 1, size=(n_rows, n_columns))
mean = 0.

for i in range(n_rows):
    for j in range(n_columns):
        mean += array[i,j]
mean = mean / (n_columns*n_rows)
print(f"The mean is: {mean}. Time needed: {time.time()-t0} seconds")
print(f"Size: {array.nbytes/1e9} Gb")
