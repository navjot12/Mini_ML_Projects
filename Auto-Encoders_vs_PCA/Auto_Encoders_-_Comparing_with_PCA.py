
# coding: utf-8

# In[6]:

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import keras
from keras.layers import Dense, Activation, Input
from keras.models import Model


# In[10]:

ds = pd.read_csv('./train.csv')
data = ds.values


# In[11]:

X_data = data[:, 1:]
X_std = X_data/255.0

n_train = int(0.5*X_std.shape[0])
n_val = int(0.25*X_std.shape[0])
X_train = X_std[:n_train]
X_val = X_std[n_train:n_train+n_val]

print X_train.shape, X_val.shape


# In[12]:

inp = Input(shape = (784, ))
embedding_dim = 64

fc1 = Dense(embedding_dim)(inp)
ac1 = Activation('tanh')(fc1)

fc2 = Dense(784)(ac1)
ac2 = Activation('sigmoid')(fc2)

autoencoder = Model(input = inp, output = ac2)

encoder = Model(input = inp, output = ac1)

dec_inp = Input(shape=(embedding_dim,))
x = autoencoder.layers[3](dec_inp)
x = autoencoder.layers[4](x)

decoder = Model(input=dec_inp, output=x)


# In[25]:

autoencoder.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])

hist = autoencoder.fit(X_train, X_train, nb_epoch=50, batch_size=100, shuffle=True, validation_data=(X_val, X_val))


# In[17]:

auto_encoder_encodes = encoder.predict(X_train)
auto_encoder_decodes = decoder.predict(auto_encoder_encodes)


# In[ ]:

from sklearn.decomposition import PCA

pca = PCA(n_components=64)
pca_dim_reducts = pca.fit_transform(X_std[:(n_train + n_val)])

pca_regenerations = pca.inverse_transform(pca_dim_reducts)


# In[1]:

plt.figure(0)
for ix in range(5, 10):
    plt.subplot(5, 3, ((ix-5) * 3) + 1)
    plt.title('Original')
    plt.imshow(X_train[ix].reshape((28, 28)), cmap='gray')
    plt.subplot(5, 3, ((ix-5) * 3) + 2)
    plt.title('A-E Regen.')
    plt.imshow(auto_encoder_decodes[ix].reshape((28, 28)), cmap='gray')
    plt.subplot(5, 3, ((ix-5) * 3) + 3)
    plt.title('PCA Regen.')
    plt.imshow(pca_regenerations[ix].reshape((28, 28)), cmap='gray')
plt.show()


# In[ ]:



