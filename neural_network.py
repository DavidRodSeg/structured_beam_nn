import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


times = 1000
mask_dim = 128
diff_dim = 2*mask_dim
mask_size = mask_dim**2
diff_size = diff_dim**2
ltrain = int( 0.7*( times ) )

# Loading the dataset
print("Loading the dataset...")
df = pd.read_csv(f"{times}_diffraction_intensity.csv", sep=";", header=None, index_col=False)
print("Done.")

# X-y split (and converting the dataframe to numpy arrays)
X_train, y_train, X_test, y_test = df.values[:ltrain,:diff_size], df.values[:ltrain,diff_size:], df.values[ltrain:,:diff_size], df.values[ltrain:,diff_size:]

# Image preprocessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

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
    tf.keras.layers.Dense(mask_size)
])
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
print(model.summary())

# Hyperparameters
n_epochs = 10
batch_size = 64

# Training the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_data=(X_test, y_test))

# Testing the model
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