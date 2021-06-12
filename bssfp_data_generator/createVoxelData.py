import os 
import numpy as np
import matplotlib.pyplot as plt
import random
from bssfp import bssfp, add_noise_gaussian
from phantom_joint import get_phantom
from phantom_brainweb import mr_brain_web_phantom, brain_web_loader,offres_gen

# Function to generate bSSFP data
def generate_data(brain_model, totalSet, numPoints):
    # Initialise parameters
    N = 128 # NxN resolution, 
    npcs = 6 # npcs = number of phase-cycle
    alpha = np.deg2rad(30) # alpha = flip angle
    TR = 3e-3 # repetition time 
    pcs = np.linspace(0, 2*np.pi, npcs, endpoint=False)
    dataSize = totalSet*numPoints # Total number of data size

    # Create empty arrays/lists to hold training data, ground truth and snr for each tissue
    voxel_data = np.zeros((dataSize, npcs), dtype=complex)
    gt_data = np.zeros((dataSize, 1))
    snr_gm  = []
    snr_wm  = []

    # Loop to create training data for voxelwise regression
    for i in range(1,totalSet+1):
        # Randomize frequency, check inhomogeneity range
        freq = 300 * random.uniform(0,1)
        offres = offres_gen(N,f=freq, rotate=True, deform=True) 

        # Create brain phantom
        phantom = mr_brain_web_phantom(brain_model,alpha=alpha,B0=3,M0=1,offres=offres)

        # Get phantom parameter
        M0, T1, T2, flip_angle, df, _ = get_phantom(phantom)

        # Simulate bSSFP acquisition with linear off-resonance
        sig = bssfp(T1, T2, TR, flip_angle, field_map=df, phase_cyc=pcs, M0=M0)

        # Add zero mean Gaussian noise with sigma = std
        #noise_level = random.uniform(0.02,0.01) # low snr
        #noise_level = random.uniform(0.006,0.002) # medium snr
        noise_level = random.uniform(0.0016,0.0012) # high snr
        #noise_level = 0.02
        sig_noise = add_noise_gaussian(sig, sigma=noise_level)

        # Calculate SNR and append to empty list of SNR
        sig_noise2 = np.sum(np.abs(sig_noise),axis=0)/npcs
        noise_r = np.sqrt((2 - (np.pi / 2)) * noise_level ** 2) # Rayleigh-corrected noise level
        snr_gm.append(np.sum(sig_noise2[np.where(T2 == 0.1)])/(noise_r*2235))
        snr_wm.append(np.sum(sig_noise2[np.where(T2 == 0.08)])/(noise_r*2059))

        # Store bSSFP data into empty voxel and ground truth arrays 
        for j in range(1,numPoints+1):
            x = random.randint(0, N-1)
            y = random.randint(0, N-1)

            while T2[x,y] == 0 or T2[x,y] == 1.99:
                x = random.randint(0, N-1)
                y = random.randint(0, N-1)

            gt_data[(numPoints*i)-j]= T2[x,y]
            voxel_data[(numPoints*i)-j]= sig_noise[:,x,y]

    print('SNR range (GM) : %.1f - %.1f' %(min(snr_gm), max(snr_gm)))
    print('SNR range (WM) : %.1f - %.1f' %(min(snr_wm), max(snr_wm)))
    snr_data = np.array([min(snr_gm), max(snr_gm),min(snr_wm), max(snr_wm)])
    return voxel_data, gt_data, snr_data


if __name__ == '__main__':
    # Brain phantom 
    dir = '/Users/yiten/Documents/MRI Relaxometry/BrainWeb' #Change to your directory of BrainWeb
    #dir = '/Users/User/Documents/MRI Relaxometry/BrainWeb' #Change to your directory of BrainWeb
    brain_model =  brain_web_loader(dir)

    #Change working directory
    os.chdir('c:\\Users\\yiten\\Documents\\FYP (Python)')
    #os.chdir('c:\\Users\\User\\Documents\\FYP-Python')

    # Initialise training data size
    totalSet = 1000 # Number of set of images
    numPoints = 100 # Number of points in a set of images
    
    # Generate training, ground truth and snr data
    voxel_data, gt_data, snr_data = generate_data(brain_model, totalSet, numPoints)

    # Save voxel, ground truth and snr arrays
    os.chdir('c:\\Users\\yiten\\Documents\\FYP (Python)\\voxel_data')
    np.save('voxel_data.npy', voxel_data)
    np.save('gt_data.npy', gt_data)
    np.save('snr_data.npy', snr_data)
    

