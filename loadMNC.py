import os
import numpy as np
import nibabel as nib
from scipy import io
import matplotlib.pyplot as plt

dir_data = '/Users/yiten/Documents/MRI Relaxometry/BrainWeb'
os.chdir(dir_data)
fileList = os.listdir()

slice = 92
mnc = [i for i in fileList if 'mnc' in i] #Simulated brain MRI in .mnc format
n_or = len(mnc)
atlas_20 = np.zeros((n_or,217,181)) #x axis, y axis

for i in range(n_or):
    img = nib.load(mnc[i])
    data = img.get_fdata()[slice,:,:].astype(int)
    
    data[np.where(data>=4)] = 0
    atlas_20[i,:,:] = np.rot90(data,k=2)
    

atlas_20 = atlas_20.astype(int)
io.savemat('atlas_20.mat', {'atlas_20': atlas_20})