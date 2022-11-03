import matplotlib.pyplot as plt
import numpy as np
import h5py

n=192
m=168

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams.update({'font.size': 18})

SVG = h5py.File('SVD.h5', 'r')
U = np.array(SVG['U'][:])

fig = plt.figure()

for i in range(6):
    ax = fig.add_subplot(231+i)
    img = ax.imshow(np.reshape(U[:,np.square(2*i)],(n,m)))
    img.set_cmap('gray')
    plt.axis('off')

plt.show()
