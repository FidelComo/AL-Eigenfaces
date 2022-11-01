import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

n=192
m=168

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams.update({'font.size': 18})

faces = h5py.File('faces_matrix.h5', 'r')['faces'][:]

avgFace = np.mean(faces,axis=1)

X = faces - np.tile(avgFace, (faces.shape[1], 1)).T
U, S, VT = np.linalg.svd(X,full_matrices=0)

hf = h5py.File("SVD.h5", 'w')

hf.create_dataset('U', data=U)
hf.create_dataset('S', data=S)
hf.create_dataset('VT', data=VT)
hf.create_dataset('avg', data=avgFace)

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
img_avg = ax1.imshow(np.reshape(avgFace,(n,m)))
img_avg.set_cmap('gray')
plt.axis('off')

ax2 = fig1.add_subplot(122)
img_u1 = ax2.imshow(np.reshape(U[:,0],(n,m)))
img_u1.set_cmap('gray')
plt.axis('off')

plt.show()