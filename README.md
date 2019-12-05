# Variational Autoencoder

In contrast to the more standard uses of neural networks as regressors or classifiers, Variational Autoencoders (VAEs) are powerful generative models, now having applications as diverse as from generating fake human faces, to producing purely synthetic music.

## How does a Variational Autoencoder work?

<p align="center">
<img src="https://github.com/crypto-code/Variational-Autoencoder/blob/master/assets/model.png"  align="middle" />   </p>

First, an encoder network turns the input samples x into two parameters in a latent space, which we will note **z_mean** and **z_log_sigma**. Then, we randomly sample similar points z from the latent normal distribution that is assumed to generate the data, via **z = z_mean + exp(z_log_sigma) * epsilon**, where epsilon is a random normal tensor. Finally, a decoder network maps these latent space points back to the original input data.

The parameters of the model are trained via two loss functions: 
* Reconstruction Loss forcing the decoded samples to match the initial inputs (just like normal autoencoders)
* KL divergence between the learned latent distribution and the prior distribution, acting as a regularization term. 

You could actually get rid of this latter term entirely, although it does help in learning well-formed latent spaces and reducing overfitting to the training data.


## Requirements:
* Python 3.6.2 (https://www.python.org/downloads/release/python-362/)
* Numpy (https://pypi.org/project/numpy/)
* Tensorflow (https://pypi.org/project/tensorflow/)
* Keras (https://pypi.org/project/Keras/)
* Scipy (https://pypi.org/project/scipy/)
* OpenCV (https://pypi.org/project/opencv-python/)

## Usage
* Download an image dataset, a pokemon dataset is already included (provided by [moxiegushi](https://github.com/moxiegushi/pokeGAN))

* Convert the Image colour scheme from RGBA to RGB by running,
```
python RGBA2RGB.py --help
usage: RGBA2RGB.py [-h] --input INPUT --output OUTPUT

Convert RGBA to RGB

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Directory containing images to resize. eg: ./resized
  --output OUTPUT  Directory to save resized images. eg: ./RGB_data
 ```
 
* Resize the image for input using the resize script,
```
python resize.py --help
usage: resize.py [-h] --input INPUT --output OUTPUT

Resize Input Images

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Directory containing images to resize. eg: ./data
  --output OUTPUT  Directory to save resized images. eg: ./resized

```


## Examples:

<p align="center">
<img src="https://github.com/crypto-code/Variational-Autoencoder/blob/master/assets/Pokemons.jpg"  align="middle" />   </p>
