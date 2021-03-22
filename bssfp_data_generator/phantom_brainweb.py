import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import nibabel as nib
import elasticdeform as ed
import random
from scipy import io, ndimage
from phantom_joint import get_phantom,show_phantom,rayleigh_correction

def mr_brain_web_phantom(bw_input, alpha, img_no=0, N=128, TR=3e-3, d_flip=10,
                         offres=None, B0=3, M0=1):
    assert img_no >= 0 and img_no < bw_input.shape[0], "Image index out of bound"

    # these are values from brain web.
    height = bw_input.shape[1]  # X
    width = bw_input.shape[2]  # Y
    dim = 6

    flip_range = np.linspace(alpha - np.deg2rad(d_flip), alpha + np.deg2rad(d_flip), N, endpoint=True)
    flip_map = np.reshape(np.tile(flip_range, N), [N, N]).transpose()
    # print(np.size(flip_range))
    # This is the default off-res map +-300Hz
    if offres is None:
        offres, _ = np.meshgrid(np.linspace(-1 / TR, 1 / TR, N), np.linspace(-1 / TR, 1 / TR, N))
    else:
        offres = offres

    sample = bw_input[img_no, :, :]

    sample = np.reshape(sample, (bw_input.shape[1], bw_input.shape[2]))
    sample = cv2.resize(sample, (N, N), interpolation=cv2.INTER_NEAREST)
    roi_mask = (sample != 0)
    ph = np.zeros((N, N, dim))

    params = _mr_relaxation_parameters(B0)

    # 0=Background, 1=CSF, 2=Gray Matter, 3=White Matter
    # ph[:,:,0] M0
    # ph[:,:,1] T1
    # ph[:,:,2] T2
    # ph[:,:,3] Flip Angle
    # ph[:,:,4] Off-res frequency

    t1_map = np.zeros((N, N))
    t2_map = np.zeros((N, N))
    t1_map[np.where(sample == 1)] = params['csf'][0]
    t1_map[np.where(sample == 2)] = params['gray-matter'][0]
    t1_map[np.where(sample == 3)] = params['white-matter'][0]
    t2_map[np.where(sample == 1)] = params['csf'][1]
    t2_map[np.where(sample == 2)] = params['gray-matter'][1]
    t2_map[np.where(sample == 3)] = params['white-matter'][1]
    #print(params['csf'][0])
    ph[:, :, 0] = M0 * roi_mask
    ph[:, :, 1] = t1_map * roi_mask
    ph[:, :, 2] = t2_map * roi_mask
    ph[:, :, 3] = flip_map * roi_mask
    ph[:, :, 4] = offres * roi_mask
    ph[:, :, 5] = sample #raw data

    #show_phantom(ph)

    return ph


def mr_brain_web_SNR(Noisy_signal,phantom, noise_level):

    _M0, _T1, _T2, _flip_angle, _df, sample = get_phantom(phantom)

    n_csf,n_gm,n_wm = 0,0,0
    for i in range(np.shape(sample)[0]):
        for j in range(np.shape(sample)[1]):
            if sample[i,j] ==1:
                n_csf+=1
            if sample[i,j] ==2:
                n_gm += 1
            if sample[i,j] ==3:
                n_wm += 1
    npcs = np.shape(Noisy_signal)[0]
    sig =np.sum(np.abs(Noisy_signal),axis=0)/npcs
    sigma = rayleigh_correction(noise_level)

    snr_csf = np.sum(sig[np.where(sample == 1)])/(sigma*n_csf)
    snr_gm = np.sum(sig[np.where(sample == 2)])/(sigma*n_gm)
    snr_wm = np.sum(sig[np.where(sample == 3)])/(sigma*n_wm)
    return snr_csf, snr_gm, snr_wm, n_csf, n_gm, n_gm

def mr_brain_web_get_sd(Noisy_signal,alpha, phantom, mapping, t1_or_t2='t1', B0=3, prec='{:.2f}', noise_level=.0):
    assert t1_or_t2 == 't1' or 't2', 'Please specify ''t1'' or ''t2'''
    params = _mr_relaxation_parameters(B0)
    M0, T1, T2, flip_angle, df, sample = get_phantom(phantom)

    # 0=Background, 1=CSF, 2=Gray Matter, 3=White Matter
    # ph[:,:,0] M0
    # ph[:,:,1] T1
    # ph[:,:,2] T2
    # ph[:,:,3] Flip Angle
    # ph[:,:,4] Off-res frequency
    if t1_or_t2 == 't1':
        t1_csf = params['csf'][0] * 1000
        t1_gm = params['gray-matter'][0] * 1000
        t1_wm = params['white-matter'][0] * 1000
    else:
        t1_csf = params['csf'][1] * 1000
        t1_gm = params['gray-matter'][1] * 1000
        t1_wm = params['white-matter'][1] * 1000

    t1_csf_mean = np.mean(mapping[np.where(sample == 1)]) * 1000
    t1_gm_mean = np.mean(mapping[np.where(sample == 2)]) * 1000
    t1_wm_mean = np.mean(mapping[np.where(sample == 3)]) * 1000
    t1_csf_std = np.std(mapping[np.where(sample == 1)]) * 1000
    t1_gm_std = np.std(mapping[np.where(sample == 2)]) * 1000
    t1_wm_std = np.std(mapping[np.where(sample == 3)]) * 1000
    # format the numbers

    t1_csf, t1_gm, t1_wm = prec.format(t1_csf), \
                           prec.format(t1_gm), \
                           prec.format(t1_wm)

    t1_csf_mean = prec.format(t1_csf_mean)
    t1_gm_mean = prec.format(t1_gm_mean)
    t1_wm_mean = prec.format(t1_wm_mean)
    t1_csf_std = prec.format(t1_csf_std)
    t1_gm_std = prec.format(t1_gm_std)
    t1_wm_std = prec.format(t1_wm_std)
    snr_csf, snr_gm, snr_wm =mr_brain_web_SNR(Noisy_signal,phantom,noise_level)
    snr_csf = prec.format(snr_csf)
    snr_gm = prec.format(snr_gm)
    snr_wm = prec.format(snr_wm)

    flip_angle = 'FA: ' + str('{:.0f}'.format(np.rad2deg(alpha))) + u'\N{DEGREE SIGN}'
    flip_angle += '|| M0: ' + str(np.amax(M0)) + ' || Noise_sd:' + str('{:.6f}'.format(rayleigh_correction(noise_level)))
    pt1 = PrettyTable()
    if t1_or_t2 == 't1':
        pt1.field_names = [flip_angle, "T1(Nominal ms)", "T1(Mean ms)", "T1(S.D. ms)",'SNR']
    else:
        pt1.field_names = [flip_angle, "T2(Nominal ms)", "T2(Mean ms)", "T2(S.D. ms)",'SNR']
    pt1.add_row(["CSF", t1_csf, t1_csf_mean, t1_csf_std,snr_csf])
    pt1.add_row(["GM", t1_gm, t1_gm_mean, t1_gm_std,snr_gm])
    pt1.add_row(["WM", t1_wm, t1_wm_mean, t1_wm_std,snr_wm])

    print(pt1)
    return


def brain_web_loader(path_data, no_of_image=1, slice_sel=175):
    assert no_of_image >= 1 and no_of_image <= 20, 'no_of_image should between 1-20'
    dir_data = path_data
    os.chdir(dir_data)
    fileList = os.listdir()

    slice = slice_sel
    mnc = [i for i in fileList if 'mnc' in i]
    n_or = no_of_image  # len(mnc)

    atlas_20 = np.zeros((n_or, 434, 362))  # axis x, axis y

    for i in range(n_or):
        img = nib.load(mnc[i])
        data = img.get_fdata()[slice, :, :].astype(int)

        data[np.where(data >= 4)] = 0
        atlas_20[i, :, :] = np.rot90(data, k=2)

    atlas_20 = atlas_20.astype(int)
    atlases = np.pad(atlas_20, ((0, 0), (39, 39), (75, 75)), 'constant')

    return atlases


def offres_gen(N, f=300, rotate=True, deform=True):
    max_rot = 360
    offres = np.zeros((N, N))
    rot_angle = max_rot * random.uniform(-1,1)
    #print('Rotational angle: '+'{:.2f}'.format(rot_angle))
    offres, _ = np.meshgrid(np.linspace(-f, f, N), np.linspace(-f, f, N))
    if rotate == True:
        offres = ndimage.rotate(offres, rot_angle, reshape=False, order=3, mode='nearest')
    if deform == True:
        offres = ed.deform_random_grid(offres, sigma=10, points=3, order=3, mode='nearest')
    return offres


def _mr_relaxation_parameters(B0):
    '''Returns MR relaxation parameters for certain tissues.

    Returns
    -------
    params : dict
        Gives entries as [A, C, (t1), t2, chi]

    Notes
    -----
    If t1 is None, the model T1 = A*B0^C will be used.  If t1 is not
    np.nan, then specified t1 will be used.
    '''

    # params['tissue-name'] = [A, C, (t1 value if explicit), t2, chi]
    # params = dict()
    # params['scalp'] = [.324, .137, np.nan, .07, -7.5e-6]
    # params['marrow'] = [.533, .088, np.nan, .05, -8.85e-6]
    # params['csf'] = [np.nan, np.nan, 4.2, 1.99, -9e-6]
    # params['blood-clot'] = [1.35, .34, np.nan, .2, -9e-6]
    # params['gray-matter'] = [.857, .376, np.nan, .1, -9e-6]
    # params['white-matter'] = [.583, .382, np.nan, .08, -9e-6]
    # params['tumor'] = [.926, .217, np.nan, .1, -9e-6]

    t1_t2 = dict()
    # t1_t2['csf'] = [4.2, 1.99] #labelled T1 and T2 map for CSF
    t1_t2['csf'] = [4.2, 0] # zero out CSF T2
    t1_t2['gray-matter'] = [.857 * (B0 ** .376), .1] #labelled T1 and T2 map for Gray Matter
    t1_t2['white-matter'] = [.583 * (B0 ** .382), .08] #labelled T1 and T2 map for White Matter
    return t1_t2


if __name__ == '__main__':
    pass
