import os 
import numpy as np
import matplotlib.pyplot as plt
from bssfp import bssfp, add_noise_gaussian
from phantom_joint import mr_joint_phantom, get_phantom
from phantom_brainweb import mr_brain_web_phantom, brain_web_loader,offres_gen
from planet import planet


if __name__ == '__main__':
    N = 128 # NxN resolution, 
    npcs = 6 # npcs = number of phase-cycle

    #Brain phantom 
    dir = '/Users/yiten/Documents/MRI Relaxometry/BrainWeb'
    data =  brain_web_loader(dir)

    #randomize frequency, check inhomogeneity range
    offres = offres_gen(N,f=1000, rotate=True, deform=True) 
    # alpha = flip angle
    alpha = np.deg2rad(30) 

    phantom = mr_brain_web_phantom(data,alpha,offres=offres)

    #Get phantom parameter
    M0, T1, T2, flip_angle,df, _sample = get_phantom(phantom)
    # print('M0:', np.mean(M0))
    # print('T1:', np.mean(T1))
    # print('T2:', np.mean(T2))

    # Simulate bSSFP acquisition with linear off-resonance
    TR = 3e-3
    pcs = np.linspace(0, 2 * np.pi, npcs, endpoint=False)
    
    sig = bssfp(T1, T2, TR, flip_angle, field_map=df, phase_cyc=pcs, M0=M0)
    print(sig.shape)
    #print(sig[0])

    #p = sig[:,64,64]
    # plt.figure(1, figsize=(12.8, 9.6))
    # plt.scatter(p.real,p.imag)
    # plt.grid(True)
    # plt.show()

    #Display phase-cycled joint/brain phantom
    plt.figure(2, figsize=(12.8, 9.6))
    for n in range(1, npcs + 1):
        plt.subplot(2, npcs / 2, n)
        plt.imshow(np.abs(sig[n - 1]))
        plt.title("PCA: %.1f" %np.rad2deg(pcs[n - 1]))
    plt.show()

    #Test PLANET algorithm
    # Meff, T1, T2 = planet(sig,alpha,TR,2,pcs,compute_df=False)
    # print(Meff)
    # print(T1)
    # print(T2)

