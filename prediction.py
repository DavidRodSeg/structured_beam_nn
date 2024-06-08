import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from pickle import load
import os


# Loading the models
current_dir = os.getcwd()
class_model = tf.keras.models.load_model(f"{current_dir}/models_and_scalers/classification_3.keras")
rectangle_model = tf.keras.models.load_model(f"{current_dir}/models_and_scalers/rectangles_pl.keras")
circles_model = tf.keras.models.load_model(f"{current_dir}/models_and_scalers/circles_pl.keras")
slits_model = tf.keras.models.load_model(f"{current_dir}/models_and_scalers/slits_pl.keras")


# Loading the scalers
rectangle_scaler = load(open(f"{current_dir}/models_and_scalers/scaler_rectangle.pkl", "rb"))
circles_scaler = load(open(f"{current_dir}/models_and_scalers/scaler_circles.pkl", "rb"))
slits_scaler = load(open(f"{current_dir}/models_and_scalers/scaler_slits.pkl", "rb"))


# Reading the example
diff_dim = 256
diff_size = diff_dim**2

current_dir = os.getcwd()
data_dir = f"{current_dir}/data_sets/modeltesting_intensity_rectangle_pl_2.csv"

df_example = pd.read_csv(data_dir, sep=";", header=None).values[:,:diff_size]

scaler = MinMaxScaler()
df_example = scaler.fit_transform(df_example.transpose()).transpose().reshape(-1, diff_dim, diff_dim, 1)


# Making the prediction
class_type = class_model.predict(df_example).transpose() # The model returns an 1-row numpy array, not a vector.
print(class_type)

if class_type[0] == 1.:

    result = rectangle_model.predict(df_example)
    result = rectangle_scaler.inverse_transform(result).transpose().astype(int)
    print(f"Class = rectangle \n a = {result[0]} \n b = {result[1]} \n p = {result[2]} \n l = {result[3]}")

elif class_type[1] == 1.:

    result = circles_model.predict(df_example).transpose()
    result = circles_scaler.inverse_transform(result).transpose().astype(int)
    print(f"Class = circle \n r = {result[0]} \n p = {result[1]} \n l = {result[2]}")

elif class_type[2] == 1.:

    result = slits_model.predict(df_example).transpose()
    result = slits_scaler.inverse_transform(result).transpose().astype(int)
    print(f"Class = slit \n n = {result[0]} \n p = {result[1]} \n l = {result[2]}")

else:

    raise TypeError("Classification error")