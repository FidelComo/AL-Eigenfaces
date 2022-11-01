import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

n=192
m=168

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams.update({'font.size': 18})

faces = h5py.File('faces_matrix.h5', 'r')['faces'][:]
persons = np.zeros((6*n,6*m))

count=0

for j in range(6):
    for k in range(6):
        persons[j*n:(j+1)*n,k*m:(k+1)*m]=np.reshape(faces[:,count*64], [n,m])
        count+=1

img = plt.imshow(persons)
img.set_cmap('gray')
plt.axis('off')
plt.show()