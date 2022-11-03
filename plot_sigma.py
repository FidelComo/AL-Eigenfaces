import matplotlib.pyplot as plt
import numpy as np
import h5py

n=192
m=168

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams.update({'font.size': 18})

SVG = h5py.File('SVD.h5', 'r')

y = np.array(SVG['S'][:])
x = np.arange(y.size)

plt.plot(x,y)
plt.yscale("log")
plt.show()

