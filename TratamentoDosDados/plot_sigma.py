import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

n=192
m=168

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams.update({'font.size': 18})

SVD = h5py.File(os.path.join('..','Matrizes','SVD.h5'), 'r')

y = np.array(SVD['S'][:])
x = np.arange(y.size)

plt.plot(x,y)
plt.yscale("log")
plt.savefig(os.path.join('..', 'Plots', 'singular_values'))
plt.show()

