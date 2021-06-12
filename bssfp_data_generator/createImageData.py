import os 
import numpy as np
import matplotlib.pyplot as plt
import random
from bssfp import bssfp, add_noise_gaussian
from phantom_joint import get_phantom
from phantom_brainweb import mr_brain_web_phantom, brain_web_loader,offres_gen, mr_brain_web_SNR

N = 128 # NxN resolution, 
npcs = 6 # npcs = number of phase-cycle
pcs = np.linspace(0, 2 * np.pi, npcs, endpoint=False)
alpha = np.deg2rad(60) # alpha = flip angle
TR = 3e-3 # repetition time

#Brain phantom 
dir = '/Users/yiten/Documents/MRI Relaxometry/BrainWeb'
ph =  brain_web_loader(dir)

#Change working directory
#os.chdir('c:\\Users\\yiten\\Documents\\FYP (Python)\\training_data')
#os.chdir('c:\\Users\\yiten\\Documents\\FYP (Python)\\training_data2')
#os.chdir('c:\\Users\\yiten\\Documents\\FYP (Python)\\test_data')

# Variables to store data
dataSize = 1000
img_data = np.zeros((dataSize, 40, 28*npcs), dtype=complex)
gt_img_data = np.zeros((dataSize, 40, 28))
snr_gm  = []
snr_wm  = []

#Loop to create training data starts here
for n in range(1,dataSize+1):
    #randomize frequency, check inhomogeneity range
    freq = 300 * random.uniform(0,1)
    offres = offres_gen(N,f=freq, rotate=True, deform=True) 

    #Create brain phantom
    phantom = mr_brain_web_phantom(ph,alpha,offres=offres)
    #Get phantom parameter
    M0, T1, T2, flip_angle,df, _sample = get_phantom(phantom)

    # Simulate bSSFP acquisition with linear off-resonance
    sig = bssfp(T1, T2, TR, flip_angle, field_map=df, phase_cyc=pcs, M0=M0)

    # Add zero mean Gaussian noise with random sigma = std
    #noise_level = random.uniform(0.02,0.0065) # low snr
    #noise_level = random.uniform(0.0045,0.0014) # medium snr
    noise_level = random.uniform(0.0011,0.00085) # high snr
    #noise_level = 0.0025
    sig_noise = add_noise_gaussian(sig, sigma=noise_level)
    _, snr_gm_temp, snr_wm_temp, _, _, _ = mr_brain_web_SNR(sig_noise, phantom, noise_level)
    snr_gm.append(snr_gm_temp)
    snr_wm.append(snr_wm_temp)

    #training_data = np.abs(sig_noise[0])
    training_data = sig_noise[0]
    training_data = training_data[43:83, 50:78]
    for i in range(1,npcs):
        training_data = np.concatenate([training_data, sig_noise[i][43:83, 50:78]], axis=1) #axis = 1 column wise
    
    img_data[n-1] = training_data

    #plt.imsave(str(n+2) + '.png',training_data)

#Loop to create ground truth starts here
#Change working directory
#os.chdir('c:\\Users\\yiten\\Documents\\FYP (Python)\\ground_truth')
# os.chdir('c:\\Users\\yiten\\Documents\\FYP (Python)\\ground_truth2')
#for n in range(1,dataSize+1):
    #Create brain phantom
    #phantom = mr_brain_web_phantom(data,alpha,offres=0)
    #Get phantom parameter
    #M0, T1, T2, flip_angle,df, _sample = get_phantom(phantom)

    gt_img_data[n-1] = T2[43:83, 50:78]
    # for i in range(1,6):
    #     ground_truth = np.concatenate([ground_truth, T2[43:83, 50:78]],axis=1) #axis = 1 column wise

    #plt.imsave(str(n) + '.png',ground_truth)

print('SNR range (GM) : %.1f - %.1f' %(min(snr_gm), max(snr_gm)))
print('SNR range (WM) : %.1f - %.1f' %(min(snr_wm), max(snr_wm)))
snr_img_data = np.array([min(snr_gm), max(snr_gm),min(snr_wm), max(snr_wm)])
os.chdir('c:\\Users\\yiten\\Documents\\MRI Relaxometry\\img_reg_data')
np.save('img_data.npy', img_data)
np.save('gt_img_data.npy', gt_img_data)
np.save('snr_img_data.npy',snr_img_data)