import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


times = 10
mask_dim = 128
diff_dim = mask_dim
mask_size = mask_dim**2
diff_size = diff_dim**2
ltrain = int( 0.7*( times*mask_dim ) )

# Loading the dataset
print("Loading the dataset...")
current_dir = os.getcwd()
relative_dir = f"{times}_diffraction_intensity.csv"
data_dir = os.path.join(current_dir, relative_dir)
df = pd.read_csv(data_dir, sep=";", header=None, index_col=False)
print("Done.")

# X-y split (and converting the dataframe to numpy arrays)
X_train, y_train, X_test, y_test = df.values[:ltrain,:diff_dim], df.values[:ltrain,diff_dim:], df.values[ltrain:,:diff_dim], df.values[ltrain:,diff_dim:]

# Image preprocessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train).reshape(-1, diff_dim, diff_dim, 1)
y_train = y_train.reshape(-1, diff_dim, diff_dim, 1)
X_test = scaler.transform(X_test).reshape(-1, diff_dim, diff_dim, 1)
y_test = y_test.reshape(-1, diff_dim, diff_dim, 1)

# Defining the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(diff_dim, diff_dim, 1)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=4, activation="relu"),
    tf.keras.layers.AveragePooling2D((2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=4, activation="relu"),
    tf.keras.layers.AveragePooling2D((2,2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(mask_dim**2),
    tf.keras.layers.Reshape((mask_dim, mask_dim, 1))
])
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
print(model.summary())

# Hyperparameters
n_epochs = 10
batch_size = 64

# Training the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_data=(X_test, y_test))

# Evaluating the model
def plot_losses():
    plt.figure(figsize=(8,4))
    plt.plot(np.log10(history.history['loss']), color='blue')
    plt.plot(np.log10(history.history['val_loss']), color='red')
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

def plot_accuracy():
    plt.figure(figsize=(8,4))
    plt.plot(np.log10(history.history['accuracy']), color='blue')
    plt.plot(np.log10(history.history['val_accuracy']), color='red')
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

plot_losses()
plot_accuracy()

# Making a prediction
current_dir = os.getcwd()
relative_dir = "modeltesting_intensity.csv"
data_dir = os.path.join(current_dir, relative_dir)
X_try = pd.read_csv(data_dir, sep=";", header=None).values[:, :mask_dim]
X_try = scaler.transform(X_try).reshape(-1, mask_dim, mask_dim, 1)

y_pred = model.predict(X_try)
y_pred = y_pred.reshape(mask_dim, mask_dim)

# Checking out the results
plt.imshow(y_pred, cmap="grey").set_clim(vmin=0, vmax=1e-5)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.colorbar()
plt.show()