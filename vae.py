from __future__ import print_function

import warnings as w
w.simplefilter(action = 'ignore')
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
import os
from PIL import Image
import cv2


batch_size = 100
original_dim = 12288
latent_dim = 2
intermediate_dim = 256
epochs = 500
epsilon_std = 1.0


parser = argparse.ArgumentParser(description='Variational Autoencoder')
parser.add_argument('--input',type=str,required=True,help='Directory containing images eg: data/')
parser.add_argument('--epoch',type=int,default=500,help='No of training iterations')
parser.add_argument('--batch',type=int,default=100,help='Batch Size')
parser.add_argument('--inter_dim',type=int,default=256,help='Dimension of Intermediate Layer')
args = parser.parse_args()

epochs = args.epoch
intermediate_dim = args.inter_dim

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)




# train the VAE on MNIST digits
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = []
y_test =[]

for each in os.listdir(args.input):
    img = cv2.imread(os.path.join(args.input,each))
    np_im = np.array(img)
    y_train.append(img)
    y_test.append(img)

x_train=np.array(y_train)
x_test=np.array(y_test)

print(x_train.shape)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')

vae.summary()

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)


# build a image generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)



n = 20  
digit_size = 64
figure = np.zeros((digit_size * n, digit_size * n, 3 * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))


for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        newimage = x_decoded[0].reshape(digit_size, digit_size, 3)
        img_cv = cv2.resize(newimage,(256,256))
        cv2.imshow('Color image', img_cv)
        cv2.waitKey(0)
