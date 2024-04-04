import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


times = 1000
mask_size = 128**2
diff_size = 4*mask_size
ltrain = int( 0.7*( times ) )

# Loading the dataset
print("Loading the dataset...")
df = pd.read_csv(f"{times}_diffraction_intensity.csv", sep=";", header=None, index_col=False)#.apply(lambda x: np.complex64(x))
print("Done.")
print(df.shape)

# X-y split (and converting the dataframe to numpy arrays)
X_train, y_train, X_test, y_test = df.values[:ltrain,:diff_size], df.values[:ltrain,diff_size:], df.values[ltrain:,:diff_size], df.values[ltrain:,diff_size:]

# Defining the model
model = tf.keras.Sequential([
    # tf.keras.layers.Normalization(), # REVISAR NORMALIZACIÓN (PORQUE NO SÉ SI USAR ESTA CAPA O USAR LA DE SKLEARN, ADEMÁS DE QUE NO SÉ SI ESTAS FUNCIONES SE PUEDEN APLICAR A NÚMEROS COMPLEJOS)
    tf.keras.layers.Input(shape=X_train.shape[1]),
    tf.keras.layers.Dense(500),
    tf.keras.layers.Dense(mask_size)
])
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

# Hyperparameters
n_epochs = 1
batch_size = 10

# Training the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs)

# Evaluating the model
loss = model.evaluate(X_test, y_test)

# Making a prediction
X_try = pd.read_csv("modeltesing_intensity.csv", sep=";", header=None).values[:, :diff_size]
y_pred = model.predict(X_try)
size = int(np.sqrt(mask_size))
y_pred = y_pred.reshape(size, size)

# Checking out the results
plt.imshow(y_pred, cmap="grey").set_clim(vmin=0, vmax=1e-5)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.colorbar()
plt.show()