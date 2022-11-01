from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

n=192
m=168

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams.update({'font.size': 18})

faces = h5py.File('faces_matrix.h5', 'r')['faces'][:]

SVG = h5py.File('SVD.h5', 'r')
U = np.array(SVG['U'][:])
avgFace = np.reshape(np.array(SVG['avg'][:]), -1)

image = Image.open('eu.pgm')
x = np.reshape(np.asarray(image), -1) - avgFace

r = [25, 50, 100, 200, 400, 800, 1600]

for i in r:
    alpha = U[:,:i].T @ x
    reconFace = avgFace + U[:,:i] @ alpha
    img = plt.imshow(np.reshape(reconFace, (n,m)))
    img.set_cmap('gray')
    plt.title('r = ' + str(i))
    plt.axis('off')
    plt.show()