import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
from pickle import dump

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

times = 500
diff_dim = 256
diff_size = diff_dim**2
ltrain = int( 0.9*( times ) )

# Loading the dataset
print("Loading the dataset...")
current_dir = os.getcwd()
relative_dir = f"{times}_diffraction_intensity_circles_pl.csv"
# relative_dir = "classification_3.csv"
data_dir = os.path.join(current_dir, relative_dir)
df = pd.read_csv(data_dir, sep=";", header=None, index_col=False)
print("Done.")

# X-y split (and converting the dataframe to numpy arrays)
X, y = df.values[:,:diff_size], df.values[:,diff_size:]

# Image preprocessing
scaler = MinMaxScaler()
X = scaler.fit_transform(np.transpose(X)).transpose() # MinMaxScaler operates for columns, thus to scale the image (which are rows in the data set) we need to transpose the df (then the rows are converted in columns). After that we transpose again the df
scalery = MinMaxScaler()
scalery.fit(y)
y = scalery.transform(y)

print(f"X max: {np.max(X)}")
print(f"Y max: {np.max(y)}")
print(f"X min: {np.min(X)}")
print(f"Y min: {np.min(y)}")

X_train, X_test = X[:ltrain,:].reshape(-1, diff_dim, diff_dim, 1), X[ltrain:,:].reshape(-1, diff_dim, diff_dim, 1)
y_train, y_test = y[:ltrain,:], y[ltrain:,:]

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Defining the model
# # Approximation model
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

    tf.keras.layers.Dense(3)
])

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


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])
print(model.summary())
param = model.count_params()

# Hyperparameters
n_epochs = 500
batch_size = 8

# Training the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_data=(X_test, y_test), shuffle=True)

# Saving the model and the scalers
current_dir = os.getcwd()
relative_dir = f"{times}_model_{param}_parameters_circles_pl.keras"
# relative_dir = "classification_3.keras"
data_dir = os.path.join(current_dir, relative_dir)
model.save(data_dir)

dump(scalery, open("scaler_circles.pkl","wb"))

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

# Loading the neural network
# model = tf.keras.models.load_model("classification.keras")

# # Making a prediction
current_dir = os.getcwd()
relative_dir = "modeltesting_intensity_rectangle_pl.csv"
data_dir = os.path.join(current_dir, relative_dir)

df_try = pd.read_csv(data_dir, sep=";", header=None)
X_try, y_try = df_try.values[:, :diff_size], df_try.values[:, diff_size:]

scaler = MinMaxScaler()
X_try = scaler.fit_transform(X_try.transpose()).transpose().reshape(-1, diff_dim, diff_dim, 1)
y_pred = model.predict(X_try)

print(f"True values: {y_try}")
print(f"Predicted values: {scalery.inverse_transform(y_pred)}")
# print(f"Predicted values: {y_pred}")


current_dir = os.getcwd()
relative_dir = "modeltesting_intensity_rectangle_2_pl.csv"
data_dir = os.path.join(current_dir, relative_dir)

df_try = pd.read_csv(data_dir, sep=";", header=None)
X_try, y_try = df_try.values[:, :diff_size], df_try.values[:, diff_size:]

scaler = MinMaxScaler()
X_try = scaler.fit_transform(X_try.transpose()).transpose().reshape(-1, diff_dim, diff_dim, 1)
y_pred = model.predict(X_try)

print(f"True values: {y_try}")
print(f"Predicted values: {scalery.inverse_transform(y_pred)}")
# print(f"Predicted values: {y_pred}")


y_pred = model.predict(X_train)

index = 75
while index < 125:

    print(f"True values: {scalery.inverse_transform(y_train[index].reshape(1,-1))}")
    # print(f"True values: {y_train[index]}")
    print(f"Predicted values: {scalery.inverse_transform(y_pred[index].reshape(1,-1))}")
    # print(f"Predicted values: {y_train[index]}")


    index = index + 1