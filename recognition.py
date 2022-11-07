import matplotlib.pyplot as plt
from crop_img import crop
import h5py
import numpy as np
import os


n=192
m=168
r=1600

SVD = h5py.File(os.path.join('Matrizes','SVD.h5'), 'r')
U = SVD['U'][:]
avgFace = np.reshape(SVD['avg'][:], -1)

def get_faces(name):
    faces = [np.reshape(crop(os.path.join('Photos', name, i), 'haarcascade_frontalface_default.xml'), -1) for i in os.listdir(os.path.join('Photos', name))]
    if faces:
        return U[:, 5:5+r].T @ np.vstack([i - avgFace for i in faces]).T
    else:
        pass

recognize = get_faces('Recognize')

persons_list = {}
for i in os.listdir(os.path.join('Photos','Persons')):
    persons_list[i] = get_faces(os.path.join('Persons',i))

distance = {}
for key, value in persons_list.items():
    distance[key] = np.apply_along_axis(lambda x: np.linalg.norm(x),0,value-recognize[:,:1]).min()

print(min(distance, key=distance.get))

modes = [1,2]
plt.plot(persons_list['Person_1'][modes[0],:],persons_list['Person_1'][modes[1],:],'d',color='k',label='Pessoa 1')
plt.plot(persons_list['Person_2'][modes[0],:],persons_list['Person_2'][modes[1],:],'^',color='r',label='Pessoa 2')
plt.plot(recognize[:,:1][modes[0],:],recognize[:,:1][modes[1],:],'^',color='b',label='?')



plt.legend()
plt.show()