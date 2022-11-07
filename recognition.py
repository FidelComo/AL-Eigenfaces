import matplotlib.pyplot as plt
from crop_img import crop
import h5py
import numpy as np
import os


n=192
m=168
r=2

SVD = h5py.File(os.path.join('Matrizes','SVD.h5'), 'r')
U = SVD['U'][:]
avgFace = np.reshape(SVD['avg'][:], -1)

def get_faces(name):
    faces = [np.reshape(crop(os.path.join('Photos', name, i), 'haarcascade_frontalface_default.xml'), -1) for i in os.listdir(os.path.join('Photos', name))]
    if faces:
        return U[:, 6:6+r].T @ np.vstack([i - avgFace for i in faces]).T
    else:
        pass

recognize = get_faces('Recognize')

persons_list = {}
for i in os.listdir(os.path.join('Photos','Persons')):
    persons_list[i] = get_faces(os.path.join('Persons',i))

distances = []

for col in recognize.T:
    distance = {}
    for key, value in persons_list.items():
        distance[key] = np.apply_along_axis(lambda x: np.linalg.norm(x-col),0,value).min()
    distances.append(distance)

for distance in distances:
    print(min(distance, key=distance.get))


modes = [0,1]
plt.plot(persons_list['ben_afflek'][modes[0],:],persons_list['ben_afflek'][modes[1],:],'d',color='k',label='ben_afflek')
plt.plot(persons_list['elton_john'][modes[0],:],persons_list['elton_john'][modes[1],:],'^',color='r',label='elton_john')
plt.plot(recognize[modes[0],0:2],recognize[modes[1],0:2],'d',color='b',label='?')
plt.plot(recognize[modes[0],2:4],recognize[modes[1],2:4],'^',color='b',label='?')

plt.legend()
plt.show()