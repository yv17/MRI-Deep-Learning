import os
import numpy as np
from scipy import io, ndimage
from skimage import transform as tf
import matplotlib.pyplot as plt
import elasticdeform as ed
import cv2
import multiprocessing as mp
from tqdm import tqdm

# Load dataset
dir_codigo = os.getcwd()
dir_datos = '/Users/jasonzhao/Brain_web/Brain_web_data_original'
dir_aumentados = '/Users/jasonzhao/Brain_web/Data_augm'

atlases = io.loadmat(dir_datos + '/atlas_20.mat')['atlas_20']
atlases = np.pad(atlases,((0,0),(39,39),(75,75)),'constant')
atlases_reflejo = np.zeros(atlases.shape)
for i in range(atlases.shape[0]):
    atlases_reflejo[i,:,:] = np.fliplr(atlases[i,:,:])
atlases = np.concatenate((atlases,atlases_reflejo), axis=0)
n_or = atlases.shape[0]
n_pixel = atlases.shape[1]

# Set generation parameters
max_rot = 30
n_tran = 2500
n_pixel_br = 128
n_div = 1000


# Start generation of new atlses
print('Num. de atlases (con reflejos): ' + str(n_or))
print('Transformaciones por atlas: ' + str(n_tran))
print('Generando atlases...')

n_arch = 0
n_atlas_gen = 0
atlas_aug = np.zeros((n_div,n_pixel_br,n_pixel_br),dtype=np.int32)

for i in range(n_or):
    print('Atlas num. ' + str(i))
    im_or = atlases[i,:,:]
    
    # Center brain
    mascara = np.nonzero(im_or>0)
    caja = np.array([[np.min(mascara[0]), np.max(mascara[0])], [np.min(mascara[1]), np.max(mascara[1])]])
    centro = np.round(np.mean(caja,axis=1))    
    centrado = np.array([256,256])-centro
    im_or = ndimage.shift(im_or, centrado, order = 0) 
    
    # Set random rotation angles
    angulo = max_rot*np.random.uniform(-1,1,size=(n_tran))    
    
    for h in tqdm(range(n_tran)):
        # Apply rotation & elastic deformation
        im_rot = ndimage.rotate(im_or,angulo[h],reshape=False,order=0)
        im_ed = ed.deform_random_grid(im_rot, sigma = 7, points = 5, order = 0)
        
        # Apply translation
        mascara = np.nonzero(im_ed>0)
        margen_v = (np.min(mascara[0]), n_pixel - np.max(mascara[0]))
        margen_h = (np.min(mascara[1]), n_pixel - np.max(mascara[1]))
        trasl = np.round(np.array([np.random.uniform(-margen_v[0],margen_v[1]),np.random.uniform(-margen_h[0],margen_h[1])]))
        im_trasl = ndimage.shift(im_ed, trasl, order = 0)
        
        atlas_aug[n_atlas_gen,:,:] = cv2.resize(im_trasl, (n_pixel_br,n_pixel_br), interpolation = cv2.INTER_NEAREST)
        n_atlas_gen += 1
        
        # Save atlases
        if n_atlas_gen == n_div:
            io.savemat(dir_aumentados + '/atlas_aug_' + str(n_arch) + '.mat', {'atlas_aug': atlas_aug})
            atlas_aug = np.zeros((n_div,n_pixel_br,n_pixel_br),dtype=np.int32)
            
            n_arch += 1            
            n_atlas_gen = 0
