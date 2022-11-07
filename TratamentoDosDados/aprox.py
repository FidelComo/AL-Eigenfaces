from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

n=192
m=168

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams.update({'font.size': 18})

faces = h5py.File(os.path.join('..','Matrizes','faces_matrix.h5'), 'r')['faces'][:]

SVD = h5py.File(os.path.join('..','Matrizes','SVD.h5'), 'r')
U = np.array(SVD['U'][:])
avgFace = np.reshape(np.array(SVD['avg'][:]), -1)

image = Image.open('img.pgm')
x = np.reshape(np.asarray(image), -1) - avgFace

r = [25, 50, 100, 200, 400, 800, 1600]

fig = plt.figure()

c=1
for i in r:
    alpha = U[:,:i].T @ x
    reconFace = avgFace + U[:,:i] @ alpha
    ax = fig.add_subplot(241+c)
    img = ax.imshow(np.reshape(reconFace, (n,m)))
    img.set_cmap('gray')
    plt.title('r = ' + str(i))
    plt.axis('off')
    c += 1

ax = fig.add_subplot(241)
img = ax.imshow(np.reshape(x, (n,m)))
img.set_cmap('gray')
plt.title('original')
plt.axis('off')

plt.show()