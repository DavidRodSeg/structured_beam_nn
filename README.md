# Study of Structural Light and its Diffraction using Machine Learning
> ⚠️ **Warning**: This repository has been archived and was created some time ago, so it may not be as optimized as it should be. Moreover, since the TFG deadline has passed, the code has not been modified. Only this *README.md* file, the *requirements.txt* file and the license have been added. Use with caution as it probably contains errors.


## 🎯 Objective
The goal of this project is to study optical diffraction and propagation using machine learning and computer vision techniques. Specifically, the repository consists of two main parts:
1. **Dataset generation**, which includes scripts for generating structured light beams (LG, HG and BG) and simulating their propagation.
2. **Neural network training and testing** for predicting certain parameters of the applied mask from the intensity patterns generated by diffraction.

Additionally, the repository also contains scripts for the processing of experimental images obtained in the laboratory.The trained neural network was applied to this real data, which was collected by passing a laser beam through a spatial light modulator (SLM) and analyzing the resulting diffraction pattern on a screen placed at a certain distance.

## 📚 Theoretical background
For clarity, this section provides a brief introduction to the physics and machine learning concepts used in the project.
### Structured light
Structured light refers to the control of light in all its degrees of freedom. This means manipulating its polarization, phase or intensity to create light beams not commonly found in nature.

A particular case of structured light are *transverse electric and magnetic* (TEM) light beams. These are solutions to the *Helmholtz equation*, a particular case of the general wave equation applicable when the *paraxial approximation* is fulfilled:

$$
\left|\frac{\partial^2 \psi}{\partial z^2}\right| \ll \left|k \cdot \frac{\partial \psi}{\partial z}\right| 
$$

Under this approximation, the equation is reduced to the *paraxial equation*:
$$
\frac{\partial^2 \psi}{\partial t^2} - 2i \cdot k \cdot \frac{\partial \psi}{\partial z} = 0
$$

The solutions to this equation are waves with electromagnetic fields perpendicular to the direction of propagation. In this project, we use some of them, including:
* **Gaussian beam**: Characterized for their gaussian intensity distribution profile.
* **Laguerre-Gauss (LG) beams**: They can be described as gaussian beams modulated by *Laguerre asssociated polynomials*, $L^p_l(x)$.
* **Hermite-Gauss (HG) beams**: Solutions to the paraxial equation in cartesian coordinates, different to the polar coordinates used in the other beams. Modulated by Hermite polynomials, $H_m$.
* **Bessel-Gauss (BG) beams**: Similar to gaussian beams in the sense that they also have axial symmetry.

### Spatial Light Modulator
Structured light was applied through a mask in both theoretical and experimental setups, as the aim of the work was to study the viability of using machine learning for extracting properties of the applied mask.

The spatial light modulator, commonly referred as SLM, was simulated by applying a complex array to the structured light beams. This was done using the Hadamart product (also called element-wise multiplication).

### Diffraction and propagation. Fraunhoffer diffraction
Diffraction is a phenomenon where the wavefront of a light beam is modified as a consequence of propagation and the superpostion principle. Typically, diffraction is studied using the *Huygens-Fresnel Principle*. According to this principle, each point on the wavefront generates a secondary wave, which can be considered a perturbation of the original wave and produces a spherical wave with the same characteristics. These secondary waves generate other tertiary waves, and so on. Mathematically, this is expressed as:

$$
U(P) = \frac{A e^{ik r_0}}{r_0} \cdot \iint_{w} \frac{e^{ik s}}{s} \cdot K(\chi) \, d\sigma
$$

With this equation, one can explain propagation and diffraction. The special case used in this project is *Fraunhoffer diffraction*, which describes the propagation to infinity of a light beam that has been modificated or diffracted by a mask. The expression of Fraunhoffer diffraction is:

$$
U(P) = -\frac{i \cos \delta}{\lambda} \cdot \frac{A e^{ik(r' + s')}}{r' s'} \cdot \iint_{A} U(\xi, \nu) \cdot e^{ik f(\xi, \eta)} \, d\xi \, d\eta
$$

where f is defined as

$$
f(\xi, \eta) = (l_0 - l)\xi + (m_0 - m)\eta + \frac{1}{2} \left( \frac{1}{r'} + \frac{1}{s'} \right) (\xi^2 + \eta^2)
$$

This approximation was chosen because of its simplicity and because its conditions are easily met in the laboratory.

### Neural networks
An *artificial neural network* (ANN) is a mathematical model inspired by the neural networks find in our brain that aims to reproduce patterns present in data. This data can take many forms, such us sounds, numerical data or images, as is in the case here. ANNs are built using small units called neurons or perceptrons, which are functions with certain weights and thresholds that produce an output based on a given input. The weights in these neurons are optimized so that the ANN can effectively detect patterns in the data.

The models used in this repository are simple neural networks based on common *multilayer* and *convolutional models*. Convolutional models are particularly useful when working with images and are based on kernels, which are arrays of size $n \times m$, that move across the image extracting information and patterns. Additionally, these models can also reduce dimensionality using *pooling layers* and not just using convolutionals.

## 🧪 Model
Two models were considered depending on the output:
* **Image output**
```ptyhon
model = tf.keras.Sequential([
tf.keras.layers.Input(shape=(diff_dim, diff_dim, 1)),
tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation="relu"),
tf.keras.layers.AveragePooling2D((2,2)),
tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation="relu"),
tf.keras.layers.AveragePooling2D((2,2)),
tf.keras.layers.Conv2D(filters=8, kernel_size=4, activation="relu"),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation="relu"),
tf.keras.layers.Dense(mask_dim**2, activation="sigmoid"),
tf.keras.layers.Reshape((mask_dim, mask_dim, 1))
])
```
* **Parameters output**
```ptyhon
model = tf.keras.Sequential([
tf.keras.layers.Input(shape=(diff_dim, diff_dim, 1)),
tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation="relu"),
tf.keras.layers.AveragePooling2D((2,2)),
tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation="relu"),
tf.keras.layers.AveragePooling2D((2,2)),
tf.keras.layers.Conv2D(filters=8, kernel_size=4, activation="relu"),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation="relu"),
tf.keras.layers.Dense(3)
])
```
The basic structure in both cases consisted of three convolutional layers and two dense layers. The first three aim to extract information from the diffraction image, while the last two reduce dimensionality and adjust the output size.

## 🛠️ How to use it
### 📦 Dependencies
This project uses Python 3.10.6. The main libraries are:
* NumPy 2.1.3
* Tensorflow 2.19.0
* Pandas 2.2.3
* OpenCV 4.10.0

However, other libraries might be needed. They can be imported using the _requirements.txt_ file.

### 📊 Dataset generation
Training and testing images are generated using *data_set_generation.py*.
```python
python .\data_set_generation.py 
```

### 🏋️ Training and testing
Training and testing are performed using neural_network.py. You can run the process with the following command:
```python
python .\neural_network.py
```

### 🔬 Experimental images preprocessing
The preprocessing of experimental images, to make them suitable for comparison with the theoretical results, is performed using image_preprocessing.py.
```python
python .\image_preprocessing.py
```
For testing the models with the experimental images use *prediction.py*.
```python
python .\prediction.py
```
You may need to delete or modified some code, as the code is not command-line friendly or correctly optimized.

### 📈 Results
Some of the results are shown in the following images:

<table>
  <tr>
    <td align="center">
      <img src="https://drive.google.com/uc?id=1EUlzEUMQwv5_2jwFMHJi1a3DjWvknGPm" width="200" /><br>
      <i>Figure 1: Experimental image.</i>
    </td>
    <td align="center">
      <img src="https://drive.google.com/uc?id=1Pt0YnKMrj5y1idAuBGs_8Mbr1Asb8BCr" width="200" /><br>
      <i>Figure 2: Beam simulation.</i>
    </td>
    <td align="center">
      <img src="https://drive.google.com/uc?id=1XQltJ01ZIGZvU4c1gGC56CDsEB4SpXgS" width="200" /><br>
      <i>Figure 3: Mask prediction.</i>
    </td>
  </tr>
</table>


## 📖 Documentation
The full TFG documentation is available in the University of Salamanca's institutional repository, GRIAL. You can access it [here](http://hdl.handle.net/10366/164446).

## 📚 References
<div id="refs" class="references csl-bib-body hanging-indent">

  <div id="ref-verdeyen1995" class="csl-entry">
    Verdeyen, J. T. 1995. "Laser Electronics". Prentice Hall.<br><br>
  </div>

  <div id="ref-oshea2015" class="csl-entry">
    O’Shea, K., and R. Nash. 2015. “An Introduction to Convolutional Neural Networks.”<br>
    <a href="https://arxiv.org/pdf/1511.08458">https://arxiv.org/pdf/1511.08458</a>.
  </div>

</div>
