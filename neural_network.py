import tensorflow as tf
import numpy as np
import pandas as pd


times = 1000
mask_size = 128**2
diff_size = 4*mask_size
ltrain = int( 0.7*( mask_size+diff_size ) )

# Loading the dataset
df = pd.read_csv(f"{times}_diffraction.csv", sep=",", header=None)

# X-y split (and converting the dataframe to numpy arrays)
X_train, y_train, X_test, y_test = df.values[:ltrain,:mask_size], df.values[:ltrain,mask_size:], df.values[ltrain:,:mask_size], df.values[ltrain:,mask_size:]

# Defining the model
model = tf.keras.Sequential(
    tf.keras.layers.Normalization(), # REVISAR NORMALIZACIÓN (PORQUE NO SÉ SI USAR ESTA CAPA O USAR LA DE SKLEARN, ADEMÁS DE QUE NO SÉ SI ESTAS FUNCIONES SE PUEDEN APLICAR A NÚMEROS COMPLEJOS)
    tf.keras.layers.Input(shape=X_train.shape[0]),
    tf.keras.layers.Dense(500, ),
)
