import os 
import numpy as np
import matplotlib.pyplot as plt
import random
from bssfp import bssfp, add_noise_gaussian
from phantom_joint import get_phantom
from phantom_brainweb import mr_brain_web_phantom, brain_web_loader,offres_gen

N = 128 # NxN resolution, 
npcs = 6 # npcs = number of phase-cycle
alpha = np.deg2rad(30) # alpha = flip angle
TR = 3e-3
pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)

#Brain phantom 
dir = '/Users/yiten/Documents/MRI Relaxometry/BrainWeb' #Change to your directory of BrainWeb
#dir = '/Users/User/Documents/MRI Relaxometry/BrainWeb' #Change to your directory of BrainWeb
data =  brain_web_loader(dir)

#Change working directory
os.chdir('c:\\Users\\yiten\\Documents\\FYP (Python)')
#os.chdir('c:\\Users\\User\\Documents\\FYP-Python')

# Initialise training data size
totalSet = 100 # Number of set of images
numPoints = 100 # Number of points in a set of images
dataSize = totalSet*numPoints # Total number of data size

# Create empty arrays to hold training data and ground truth
voxel_data = np.zeros((dataSize, npcs), dtype=np.complex)
gt_data = np.zeros((dataSize, 1))

# Loop to create training data for voxelwise regression
for i in range(1,totalSet+1):
    # Randomize frequency, check inhomogeneity range
    # freq = 1000 * random.uniform(0,1)
    # offres = offres_gen(N,f=freq, rotate=True, deform=True) 

    # Create brain phantom
    #phantom = mr_brain_web_phantom(data,alpha,offres=offres)
    phantom = mr_brain_web_phantom(data,alpha,B0=3,M0=1,offres=offres_gen(N,f=300))

    # Get phantom parameter
    M0, T1, T2, flip_angle, df, _ = get_phantom(phantom)

    # Simulate bSSFP acquisition with linear off-resonance
    
    sig = bssfp(T1, T2, TR, flip_angle, field_map=df, phase_cyc=pcs, M0=M0)

    # Add zero mean Gaussian noise with sigma = std
    noise_level = 0.001
    sig_noise = add_noise_gaussian(sig, sigma=noise_level)

    for j in range(1,numPoints+1):
        x = random.randint(0, N-1)
        y = random.randint(0, N-1)

        while T2[x,y] == 0:
            x = random.randint(0, N-1)
            y = random.randint(0, N-1)

        gt_data[(numPoints*i)-j]= T2[x,y]
        voxel_data[(numPoints*i)-j]= sig_noise[:,x,y]

# Save voxel and ground truth data numpy array
np.save('voxel_train.npy', voxel_data)
np.save('gt_train.npy', gt_data)
