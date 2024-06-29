import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from pickle import load
import os
import time


diff_dim = 256
diff_size = diff_dim**2

# Loading the models
class_model = tf.keras.models.load_model("categorical.keras")
rectangle_model = tf.keras.models.load_model("rectangular_parameters.keras")
circles_model = tf.keras.models.load_model("circular_parameters.keras")
slits_model = tf.keras.models.load_model("slits_parameters.keras")
all_model = tf.keras.models.load_model("all_parameters.keras")

# Loading the scalers
rectangle_scaler = load(open("rectangular_parameters.pkl", "rb"))
circles_scaler = load(open("circular_parameters.pkl", "rb"))
slits_scaler = load(open("slits_parameters.pkl", "rb"))
all_scaler = load(open("all_parameters.pkl", "rb"))

# Reading the example
# data_dir = "preprocessed_images/preprocessed_images.csv"
data_dir = "buenas/buenas.csv"

df_example = pd.read_csv(data_dir, sep=";", header=None).values[:,:diff_size].reshape(-1, diff_size)

scaler = MinMaxScaler()
df_example = scaler.fit_transform(df_example.transpose()).transpose()

# Making the prediction
def prediction_multiple(vector):
    vector = vector.reshape(-1, diff_dim, diff_dim, 1)
    class_type = class_model.predict(vector).transpose() # The model returns an 1-row numpy array, not a vector.

    if class_type[0] == 1.:

        result = rectangle_model.predict(vector)
        result = rectangle_scaler.inverse_transform(result).transpose().astype(int)
        print(f"Class = rectangle \n a = {result[0]} \n b = {result[1]} \n ind = {result[2]}")

    elif class_type[1] == 1.:

        result = circles_model.predict(vector)
        result = circles_scaler.inverse_transform(result).transpose().astype(int)
        print(f"Class = circle \n r = {result[0]} \n p = {result[1]} \n ind = {result[2]}")

    elif class_type[2] == 1.:

        result = slits_model.predict(vector)
        result = slits_scaler.inverse_transform(result).transpose().astype(int)
        print(f"Class = slit \n n = {result[0]} \n p = {result[1]} \n ind = {result[2]}")

    else:

        raise TypeError("Classification error")
    

def prediction_geometrical(vector):
    vector = vector.reshape(-1, diff_dim, diff_dim, 1)

    result = geometrical_model.predict(vector)
    result = geometrical_scaler.inverse_transform(result).transpose()

    print(f"Sides: {result[0]} \n Radius(px): {result[1]} \n Angle(º): {result[2]}")


def prediction_all(vector):
    vector = vector.reshape(-1, diff_dim, diff_dim, 1)

    result = all_model.predict(vector)
    result = all_scaler.inverse_transform(result).transpose()

    print(f"Side1(px): {result[0]} \n Side2(px): {result[1]} \n Type: {result[2]}")

t0 = time.time()
for i in range(df_example.shape[0]):
    # prediction_multiple(df_example[i, :])
    # prediction_geometrical(df_example[i, :])
    prediction_all(df_example[i, :])

print(f"Tiempo: {(time.time()-t0)/df_example.shape[0]} segundos")