import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

n=192
m=168

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams.update({'font.size': 18})

SVD = h5py.File(os.path.join('..','Matrizes','SVD.h5'), 'r')
U = np.array(SVD['U'][:])

fig = plt.figure()

for i in range(6):
    ax = fig.add_subplot(231+i)
    img = ax.imshow(np.reshape(U[:,np.square(2*i)],(n,m)))
    img.set_cmap('gray')
    plt.axis('off')

plt.savefig(os.path.join('..', 'Plots', 'eigenfaces'))
plt.show()