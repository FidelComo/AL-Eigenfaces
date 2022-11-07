import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

faces = h5py.File(os.path.join('..','Matrizes','faces_matrix.h5'), 'r')['faces'][:]

SVD = h5py.File(os.path.join('..','Matrizes','SVD.h5'), 'r')
U = np.array(SVD['U'][:])
avgFace = np.reshape(np.array(SVD['avg'][:]), -1)

P1num = 2
P2num = 7

P1 = faces[:,64*(P1num-1):64*P1num]
P2 = faces[:,64*(P2num-1):64*P2num]

P1 = P1 - np.tile(avgFace,(P1.shape[1],1)).T
P2 = P2 - np.tile(avgFace,(P2.shape[1],1)).T

PCAmodes = [5, 6]
PCACoordsP1 = U[:,PCAmodes-np.ones_like(PCAmodes)].T @ P1
PCACoordsP2 = U[:,PCAmodes-np.ones_like(PCAmodes)].T @ P2

plt.plot(PCACoordsP1[0,:],PCACoordsP1[1,:],'d',color='k',label='Pessoa 2')
plt.plot(PCACoordsP2[0,:],PCACoordsP2[1,:],'^',color='r',label='Pessoa 7')

plt.legend()
plt.savefig(os.path.join('..','Plots','PC_5-6'))
plt.show()

PCAmodes = [1, 2]
PCACoordsP1 = U[:,PCAmodes-np.ones_like(PCAmodes)].T @ P1
PCACoordsP2 = U[:,PCAmodes-np.ones_like(PCAmodes)].T @ P2

plt.plot(PCACoordsP1[0,:],PCACoordsP1[1,:],'d',color='k',label='Pessoa 2')
plt.plot(PCACoordsP2[0,:],PCACoordsP2[1,:],'^',color='r',label='Pessoa 7')

plt.legend()
plt.savefig(os.path.join('..','Plots','PC_1-2'))
plt.show()
