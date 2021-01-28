import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from phantominator import shepp_logan
from tqdm import tqdm
from bssfp import bssfp,add_noise_gaussian
from phantom_joint import mr_joint_phantom, mr_joint_get_sd, get_phantom,mr_joint_SNR
from phantom_brainweb import mr_brain_web_phantom, brain_web_loader, mr_brain_web_get_sd, offres_gen
from planet import planet
from skimage.restoration import unwrap_phase

if __name__ == '__main__':
    print('')
    # Shepp-Logan
    N, npcs, B0 = 128, 6, 3
    alpha = np.deg2rad(30)
    TR = 3e-3
    pcs = np.linspace(0, 2 * np.pi, npcs, endpoint=False)
    dflip = 5
    #critical for 30  = 0.0275\

    # 0.0088
    noise_level = 1

    dir = '/Users/yiten/Documents/MRI Relaxometry/BrainWeb'
    data =  brain_web_loader(dir)
    phantom = mr_brain_web_phantom(data,alpha,B0=B0,M0=1,offres=offres_gen(N,f=300))
    M0, T1, T2, flip_angle, df, _sample = get_phantom(phantom)
    
    # Simulate bSSFP acquisition with linear off-resonance

    sig = bssfp(T1, T2, TR, flip_angle, field_map=0, phase_cyc=pcs, M0=M0)

    #add noise
    sig_fft = np.fft.fft2(sig)
    sig_noise = np.fft.ifft2(add_noise_gaussian(sig_fft, sigma=noise_level))
    # plt.figure()
    # plt.scatter(sig_noise[:,64,64].real,sig_noise[:,64,64].imag)
    # plt.xlabel('Real')
    # plt.ylabel('Imaginary')
    # plt.title('Points of bSSFP complex signal')
    # plt.show()
    Noisy_sig = sig_noise

    # Display phase-cycled images w/o noise
    # plt.figure(1, figsize=(12.8, 9.6))
    # plt.title('Phase_Cycled images W/O noise')
    # for n in range(1, npcs + 1):
    #     plt.subplot(2, npcs / 2, n)
    #     plt.imshow(np.abs(sig[n - 1]))
    #     plt.title('PC %d Degrees' % np.rad2deg(pcs[n - 1]))
    # plt.show()

    # Display phase-cycled images with noise
    plt.figure(2, figsize=(12.8, 9.6))
    plt.title('Phase_Cycled images with noise')
    for n in range(1, npcs + 1):
        plt.subplot(2, npcs / 2, n)
        plt.imshow(np.abs(sig_noise[n - 1]))
        plt.title('PC %d Degrees' % np.rad2deg(pcs[n - 1]))
    plt.show()

    # Do T1, T2 mapping for each pixel
    mask = np.abs(M0.flatten()) > 1e-8
    idx = np.argwhere(mask).squeeze()

    sig = np.reshape(sig, (npcs, -1))
    sig_noise = np.reshape(sig_noise, (npcs, -1))

    # # Create some new arrays
    Meff = np.zeros((N * N))
    T1map = np.zeros((N * N))
    T2map = np.zeros((N * N))
    local_df = np.zeros((N * N))

    counter = 0

    for idx0 in tqdm(idx, leave=False, total=idx.size):
        Meff[idx0], T1map[idx0], T2map[idx0], local_df[idx0] = planet(sig_noise[:, idx0], alpha=alpha, TR=TR, T1_guess=2, pcs=pcs, compute_df=True)
        counter += 1
    mask = np.reshape(mask, (N, N))
    Meff = np.reshape(Meff, (N, N))
    T1map = np.reshape(T1map, (N, N))
    T2map = np.reshape(T2map, (N, N))

    local_df = np.reshape(local_df, (N, N))

    local_df = np.unwrap(local_df, discont=.5 / TR)
    local_df -= (local_df.max() - local_df.min()) / 2

    local_df *= mask

    T1_d = T1 - T1map
    T2_d = T2 - T2map

    mr_brain_web_get_sd(Noisy_sig,alpha,phantom, T1map,t1_or_t2='t1',noise_level = noise_level)
    mr_brain_web_get_sd(Noisy_sig,alpha,phantom, T2map, t1_or_t2='t2',noise_level = noise_level)
    print('Counter: ', counter)
    fig0, axs0 = plt.subplots(2, 3, figsize=(12.8, 6.4))
    dict_title={
        0: 'T1 Groundtruth',
        1: 'T1 Mapping',
        2: 'T1 Residual',
        3: 'T2 Groundtruth',
        4: 'T2 Mapping',
        5: 'T2 Residual'
    }
    dict_result={
        0: T1 * 1000,
        1: T1map * 1000,
        2: T1_d*1000,
        3: T2 * 1000,
        4: T2map * 1000,
        5: T2_d*1000
    }
    dict_alphabet={
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f'
    }
    idx1, idx2 = 0, 0
    fig0.subplots_adjust(top=1.2)
    for idx00 in range(len(dict_title)):
        axs0[idx1, idx2].set_title(dict_title.get(idx00,None), fontdict={'fontsize': 13})
        im = axs0[idx1, idx2].imshow(dict_result.get(idx00,None))
        cbar=fig0.colorbar(im, ax=axs0[idx1, idx2])
        cbar.ax.set_title('ms')
        axs0[idx1, idx2].text(.01, .99, dict_alphabet.get(idx00,None), fontdict={'fontsize': 35}, weight='bold',
                        horizontalalignment='left',color = 'w',
                        verticalalignment='top',
                        transform=axs0[idx1, idx2].transAxes)
        idx2 += 1
        if idx2 == 3:
            idx2 = 0
            idx1 += 1

    plt.show()


    plt.figure(4, figsize=(19.2, 10.8))
    nx, ny = 2, 5
    plt.subplot(nx, ny, 1)
    plt.imshow(T1 * 1000)
    plt.colorbar()
    plt.title('T1 Groundtruth')
    
    plt.subplot(nx, ny, 2)
    plt.imshow(T1map * 1000)
    plt.colorbar()
    plt.title('T1 Mapping')
    
    plt.subplot(nx, ny, 3)
    plt.imshow(T2 * 1000)
    plt.colorbar()
    plt.title('T2 Groundtruth')
    
    plt.subplot(nx, ny, 4)
    plt.imshow(T2map * 1000)
    plt.colorbar()
    plt.title('T2 Mapping')
    
    plt.subplot(nx, ny, 5)
    plt.imshow((T1 - T1map) * 1000)
    plt.colorbar()
    plt.title('T1 Difference')
    
    plt.subplot(nx, ny, 6)
    plt.imshow((T2 - T2map) * 1000)
    plt.colorbar()
    plt.title('T2 Difference')
    
    plt.subplot(nx, ny, 7)
    plt.imshow(df)
    plt.colorbar()
    plt.title('df')
    
    plt.subplot(nx, ny, 8)
    plt.imshow(local_df)
    plt.colorbar()
    plt.title('local_df')
    
    plt.subplot(nx, ny, 9)
    plt.imshow(Meff)
    plt.colorbar()
    plt.title('Meff')
    
    plt.subplot(nx, ny, 10)
    plt.imshow(mask)
    plt.colorbar()
    plt.title('MASK')
    plt.show()
