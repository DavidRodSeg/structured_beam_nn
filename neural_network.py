import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from pickle import dump
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# tf.config.set_visible_devices([], 'GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def calculate_mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)

    return mse

def load_csv_in_chunks(file_path, chunksize=1000, sep=";", header=None, index_col=False):
    chunk_list = []
    for chunk in pd.read_csv(file_path, sep=sep, header=header, index_col=index_col, chunksize=chunksize):
        chunk_list.append(chunk)
    return pd.concat(chunk_list, axis=0)

times = 1500
diff_dim = 256
diff_size = diff_dim**2
ltrain = int( 0.8*( times ) )

# ------------------- Loading the dataset -------------------
print("Loading the dataset...")
rectangular_parameters_path = "rectangular_images.csv"
circular_parameters_path = "circular_images.csv"
slits_parameters_path = "slit_images.csv"

df1 = load_csv_in_chunks(rectangular_parameters_path)
df2 = load_csv_in_chunks(circular_parameters_path)
df3 = load_csv_in_chunks(slits_parameters_path)

df = pd.concat([df1, df2, df3], ignore_index=True)
print(df.shape)

print("Done.")

# ------------------- X-y split (and converting the dataframe to numpy arrays) -------------------
X, y = df.values[:,:diff_size], df.values[:,diff_size:]
print(X.shape, y.shape)

# ------------------- Image preprocessing -------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
scalery = MinMaxScaler()
X_train = scaler.fit_transform(np.transpose(X_train)).transpose().reshape(-1, diff_dim, diff_dim, 1)
X_val = scaler.fit_transform(np.transpose(X_val)).transpose().reshape(-1, diff_dim, diff_dim, 1)
X_test = scaler.fit_transform(np.transpose(X_test)).transpose().reshape(-1, diff_dim, diff_dim, 1)
y_train = scaler.fit_transform(np.transpose(y_train)).transpose().reshape(-1, diff_dim, diff_dim, 1)
y_val = scaler.fit_transform(np.transpose(y_val)).transpose().reshape(-1, diff_dim, diff_dim, 1)
y_test = scaler.fit_transform(np.transpose(y_test)).transpose().reshape(-1, diff_dim, diff_dim, 1)
# scalery.fit(y_train)
# y_train = scalery.transform(y_train)
# y_val = scalery.transform(y_val)


# ------------------- Defining the model -------------------
# # Approximation model
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(diff_dim, diff_dim, 1)),

#     # First convolutional block
#     tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation="relu"),
#     tf.keras.layers.AveragePooling2D((2,2)),

#     # Second convolutional block
#     tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation="relu"),
#     tf.keras.layers.AveragePooling2D((2,2)),

#     # Third convolutional block
#     tf.keras.layers.Conv2D(filters=8, kernel_size=4, activation="relu"),

#     tf.keras.layers.Flatten(),

#     # Fully connected layers
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(3)
# ])

# # Classification model
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(diff_dim, diff_dim, 1)),

#     # First convolutional block
#     tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation="relu"),
#     tf.keras.layers.AveragePooling2D((2,2)),

#     # Second convolutional block
#     tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation="relu"),
#     tf.keras.layers.AveragePooling2D((2,2)),

#     # Third convolutional block
#     tf.keras.layers.Conv2D(filters=8, kernel_size=4, activation="relu"),

#     tf.keras.layers.Flatten(),

#     # Fully connected layers
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(3, activation="softmax")
# ])

# Image approximation model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(diff_dim, diff_dim, 1)),

    # First convolutional block
    tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation="relu"),
    tf.keras.layers.AveragePooling2D((2,2)),

    # Second convolutional block
    tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation="relu"),
    tf.keras.layers.AveragePooling2D((2,2)),

    # Third convolutional block
    tf.keras.layers.Conv2D(filters=8, kernel_size=4, activation="relu"),

    tf.keras.layers.Flatten(),

    # Fully connected layers
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(diff_size, activation="sigmoid"), # The sigmoid function is necessary for obtaining an amplitude mask with only two values
    tf.keras.layers.Reshape((diff_dim, diff_dim, 1))
])


optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])
print(model.summary())

# Hyperparameters
n_epochs = 300
batch_size = 8

# Training the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_data=(X_val, y_val), shuffle=True)

# Saving the model and the scalers
data_dir = "all_images.keras"
model.save(data_dir)

dump(scalery, open("all_images.pkl","wb"))

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

y_pred = model.predict(X_test)

# index = 0
# while index < 15:

#     print(f"True values: {y_test[index]}")
#     print(f"Predicted values: {scalery.inverse_transform(y_pred[index].reshape(1,-1))}")

#     index = index + 1

# print(calculate_mse(scalery.transform(y_test), y_pred))

index = 0
while index < 15:

    plt.imshow(y_test[index], cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'{index}_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.imshow(y_pred[index], cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'{index}_expected.png', dpi=300, bbox_inches='tight')
    plt.close()

    index = index + 1