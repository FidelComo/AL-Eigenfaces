from PIL import Image
import numpy as np
import os
import h5py

def img_to_vector(path):
    image = Image.open(path)
    return np.reshape(np.asarray(image), -1)

image = Image.open('CroppedYale\yaleB01\yaleB01_P00A+000E+00.pgm')

img_list = [img_to_vector(os.path.join(i,j)) for i in os.scandir('CroppedYale') for j in os.listdir(i)]

matriz = np.vstack(img_list).transpose()

hf = h5py.File("faces_matrix.h5", 'w')

hf.create_dataset('faces', data=matriz)