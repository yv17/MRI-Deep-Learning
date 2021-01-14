'''phantoms'''
# methods for generation of statistical results
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from prettytable import PrettyTable
from bssfp import add_noise_gaussian, bssfp
from planet import planet
from scipy import stats

np.random.seed(19960904)


def mr_joint_phantom(alpha,
                     T1=(1.2, 1.4, 3.6),
                     T2=(0.03, 0.05, 0.75),
                     N=128, TR=3e-3, d_flip=10,
                     offres=None, M0=1):
    '''This method simulates a numberic phantom that consisted with three tissue blocks
    Top: Cartilage
    Middle: Muscle
    Bottom: Synovial fluid
    Each tissue block is 21 pixel high and 64 pixel wide
    '''
    # Determine size of phantom
    # Dimension of the phantom
    dim = 6
    # size of the tissue block
    height, width = 21, 64
    padding_X, padding_Y = 32, 32

    if np.isscalar(N):
        X, Y = N, N
    else:
        X, Y = N[:]
    T1c, T1m, T1s = T1[0], T1[1], T1[2]
    T2c, T2m, T2s = T2[0], T2[1], T2[2]

    # This simulates flip angle inhomogeneities
    flip_range = np.linspace(alpha - np.deg2rad(d_flip), alpha + np.deg2rad(d_flip), height, endpoint=True)
    # print(np.size(flip_range))

    ph = np.zeros((X, Y, dim))

    # This is the default off-res map +-300Hz
    if offres is None:
        offres, _ = np.meshgrid(
            np.linspace(-1 / TR, 1 / TR, width),
            np.linspace(-1 / TR, 1 / TR, width))
        offres = np.pad(offres, ((32, 32), (32, 32)), 'constant')

    # ph[:,:,0] M0
    # ph[:,:,1] T1
    # ph[:,:,2] T2
    # ph[:,:,3] Flip Angle
    # ph[:,:,4] Off-res frequency
    # ph[:,:,5] Discrete model labeling 0:Background 1:Cartilage 2:Muscle 3:Synovial fluid
    # coordinates

    # Cartilage(31+height,31+width) ph[cart_Y, cart_X, 4] = offres[cart_X - padding_X]

    for cart_X in range(padding_X, padding_X + width):
        # Cartilage
        for cart_Y in range(padding_Y, padding_Y + height):
            ph[cart_Y, cart_X, 0] = M0
            ph[cart_Y, cart_X, 1] = T1c
            ph[cart_Y, cart_X, 2] = T2c
            ph[cart_Y, cart_X, 3] = flip_range[cart_Y - padding_Y]
            # ph[cart_Y, cart_X, 4] = offres
            ph[cart_Y, cart_X, 5] = 1
        # Muscle
        for cart_Y in range(padding_Y + height, padding_Y + 2 * height):
            ph[cart_Y, cart_X, 0] = M0
            ph[cart_Y, cart_X, 1] = T1m
            ph[cart_Y, cart_X, 2] = T2m
            ph[cart_Y, cart_X, 3] = flip_range[cart_Y - (padding_Y + height)]
            # ph[cart_Y, cart_X, 4] = offres[cart_X - padding_X]
            ph[cart_Y, cart_X, 5] = 2
        # Synovial Fluid
        for cart_Y in range(padding_Y + 2 * height, padding_Y + 3 * height):
            ph[cart_Y, cart_X, 0] = M0
            ph[cart_Y, cart_X, 1] = T1s
            ph[cart_Y, cart_X, 2] = T2s
            ph[cart_Y, cart_X, 3] = flip_range[cart_Y - (padding_Y + 2 * height)]
            # ph[cart_Y, cart_X, 4] = offres[cart_X - padding_X]
            ph[cart_Y, cart_X, 5] = 3

    ph[:, :, 4] = offres
    # showplot_single(ph[:, :, 1] * 1000, title='T1', cbar_title='ms')
    # showplot_single(ph[:, :, 2] * 1000, title='T2', cbar_title='ms')
    # showplot_single(np.rad2deg(ph[:, :, 3]), title='Actual Flip Angle', cbar_title='Degree (\N{DEGREE SIGN})')
    # show_phantom(ph)

    return ph


def rayleigh_correction(mu):
    '''calculation of actual mean of bivariate zero-mean Gaussian with same STD'''
    return np.sqrt((2 - (np.pi / 2)) * mu ** 2)


def mr_joint_SNR(Noisy_signal, noise_level):
    rayleigh = rayleigh_correction(noise_level)
    height, width = 21, 64
    padding_X, padding_Y = 32, 32

    cart_sig = Noisy_signal[:, padding_Y:padding_Y + height
    , padding_X:padding_X + width]

    musc_sig = Noisy_signal[:, padding_Y + height:padding_Y + 2 * height
    , padding_X:padding_X + width]

    synov_sig = Noisy_signal[:, padding_Y + 2 * height:padding_Y + 3 * height
    , padding_X:padding_X + width]
    l = len(cart_sig.flatten())

    cart_snr = np.sum(np.abs(cart_sig.flatten())) / (rayleigh * l)
    musc_snr = np.sum(np.abs(musc_sig.flatten())) / (rayleigh * l)
    synov_snr = np.sum(np.abs(synov_sig.flatten())) / (rayleigh * l)

    # cart_snr = np.sum(np.abs(cart_sig.flatten())) / (l)
    # musc_snr = np.sum(np.abs(musc_sig.flatten())) / (l)
    # synov_snr = np.sum(np.abs(synov_sig.flatten())) / (l)
    prec = '{:.2f}'
    cart_snr, musc_snr, synov_snr = prec.format(cart_snr), \
                                    prec.format(musc_snr), \
                                    prec.format(synov_snr)

    return cart_snr, musc_snr, synov_snr


def mr_joint_get_sd(Noisy_signal, noise_level, alpha, phantom, gt, mapping, npcs, dflip, TR, t1_or_t2='t1',plot=False ):
    # assert gt.shape() == (128, 128), 'Image size has to be 128*128'
    # assert mapping.shape() == (128, 128), 'Image size has to be 128*128'
    assert t1_or_t2 == 't1' or 't2', 'Please specify ''t1'' or ''t2'''

    # get phantom parameters
    M0, T1, T2, flip_map, df, _sample = get_phantom(phantom)

    # extracting each tissue block
    height, width = 21, 64
    padding_X, padding_Y = 32, 32

    cart_mapping = mapping[padding_Y:padding_Y + height
    , padding_X:padding_X + width]

    musc_mapping = mapping[padding_Y + height:padding_Y + 2 * height
    , padding_X:padding_X + width]

    synov_mapping = mapping[padding_Y + 2 * height:padding_Y + 3 * height
    , padding_X:padding_X + width]

    # Conversion to ms
    t1_cart, t1_musl, t1_synov = gt[padding_Y + 5, padding_X + 5] * 1000, \
                                 gt[padding_Y + height + 5, padding_X + 5] * 1000, \
                                 gt[padding_Y + 2 * height + 5, padding_X + 5] * 1000

    t1_cart_mean = np.mean(cart_mapping) * 1000
    t1_musl_mean = np.mean(musc_mapping) * 1000
    t1_synov_mean = np.mean(synov_mapping) * 1000

    t1_cart_std = np.std(cart_mapping) * 1000
    t1_musl_std = np.std(musc_mapping) * 1000
    t1_synov_std = np.std(synov_mapping) * 1000

    # formatting numbers to display
    prec = '{:.2f}'
    t1_cart, t1_musl, t1_synov = prec.format(t1_cart), \
                                 prec.format(t1_musl), \
                                 prec.format(t1_synov)
    t1_cart_mean = prec.format(t1_cart_mean)
    t1_musl_mean = prec.format(t1_musl_mean)
    t1_synov_mean = prec.format(t1_synov_mean)
    t1_cart_std = prec.format(t1_cart_std)
    t1_musl_std = prec.format(t1_musl_std)
    t1_synov_std = prec.format(t1_synov_std)

    cart_snr, musc_snr, synov_snr = mr_joint_SNR(Noisy_signal, noise_level)

    flip_angle = 'FA: ' + str('{:.0f}'.format(np.rad2deg(alpha))) + u'\N{DEGREE SIGN}' + u"\u00B1" + ' ' + str(
        '{:.0f}'.format(dflip)) + u'\N{DEGREE SIGN}'
    noise_level = rayleigh_correction(noise_level)
    # display results
    pt1 = PrettyTable()
    if t1_or_t2 == 't1':
        pt1.field_names = ['Noise: \u03c3=%.4f' % noise_level, "T1(Nominal ms)", "T1(Mean ms)", "T1(S.D. ms)", 'SNR',
                           'Info']
    else:
        pt1.field_names = ['Noise: \u03c3=%.4f' % noise_level, "T2(Nominal ms)", "T2(Mean ms)", "T2(S.D. ms)", 'SNR',
                           'Info']
    pt1.add_row(["Cartilage", t1_cart, t1_cart_mean, t1_cart_std, cart_snr, flip_angle])
    pt1.add_row(["Muscle", t1_musl, t1_musl_mean, t1_musl_std, musc_snr, 'No_PC: %d' % npcs])
    pt1.add_row(["Synovial fluid", t1_synov, t1_synov_mean, t1_synov_std, synov_snr, 'TR: %dms' % (TR * 1e3)])
    print(pt1)

    tissue_type = {
        0: "Cartilage",
        1: "Muscle",
        2: "Synovial fluid"
    }
    if plot ==True:
        for t in range(3):
            t_type = tissue_type.get(t, None)
            pt2 = PrettyTable()
            tissue = t_type + ' (Nominal: ' + str('{:.0f}'.format(np.rad2deg(alpha))) + u'\N{DEGREE SIGN}' + ')'
            if t1_or_t2 == 't1':
                pt2.field_names = ['No. of Rows', tissue, "T1(Nominal ms)", "T1(Mean ms)", "T1(S.D. ms)"]
            else:
                pt2.field_names = ['No. of Rows', tissue, "T2(Nominal ms)", "T2(Mean ms)", "T2(S.D. ms)"]

            if t_type == "Cartilage":
                mapping_temp = cart_mapping
                # t1_cart = t1_cart
            elif t_type == "Muscle":
                mapping_temp = musc_mapping
                t1_cart = t1_musl
            else:
                mapping_temp = synov_mapping
                t1_cart = t1_synov

            for h in range(height):
                flip = 'Flip angle: ' + str('{:.2f}'.format(np.rad2deg(flip_map[padding_Y + h
                , padding_X + 5]))) + u'\N{DEGREE SIGN}'
                t1_cart_mean = prec.format(np.mean(mapping_temp[h, :]) * 1000)
                t1_cart_std = prec.format(np.std(mapping_temp[h, :]) * 1000)
                pt2.add_row([h + 1, flip, t1_cart, t1_cart_mean, t1_cart_std])
            print(pt2)

    dict_stats = {0: {0: float(t1_cart), 1: float(t1_cart_mean), 2: float(t1_cart_std), 3: float(cart_snr)},
                  1: {0: float(t1_musl), 1: float(t1_musl_mean), 2: float(t1_musl_std), 3: float(musc_snr)},
                  2: {0: float(t1_synov), 1: float(t1_synov_mean), 2: float(t1_synov_std), 3: float(synov_snr)},
                  3: {0: 'Noise: \u03c3=%.4f' % noise_level, 1: flip_angle, 2: 'No_PC: %d' % npcs,
                      3: 'TR: %dms' % (TR * 1e3)}}

    return dict_stats


def mr_joint_montecarlo(phantom, alpha, n, npcs=8, M0=1, TR=3e-3, noise_level=.0):
    # MonteCarlo simulation with single pixel repeat n times
    # only use this method with flip angle variation d_dlip set to 0.

    pcs = np.linspace(0, 2 * np.pi, npcs, endpoint=False)
    height, width = 21, 64
    padding_X, padding_Y = 32, 32
    # Coordinates of pixel of each
    cart_x, cart_y = padding_Y + 5, padding_X + 5
    musl_x, musl_y = padding_Y + height + 5, padding_X + 5
    synov_x, synov_y = padding_Y + 2 * height + 5, padding_X + 5

    tissue_name = {
        0: "Cartilage",
        1: "Muscle",
        2: "Synovial fluid"
    }

    M0, T1, T2, flip_angle, df, _sample = get_phantom(phantom)

    sig = bssfp(T1, T2, TR, flip_angle, field_map=df, phase_cyc=pcs, M0=M0)
    sig2d = np.atleast_2d(sig)
    sig_fft = np.fft.fft2(sig2d)

    T1map_cart = np.zeros(n)
    T2map_cart = np.zeros(n)
    T1map_musl = np.zeros(n)
    T2map_musl = np.zeros(n)
    T1map_synov = np.zeros(n)
    T2map_synov = np.zeros(n)

    Meff_cart= np.zeros(n)
    Meff_musl= np.zeros(n)
    Meff_synov= np.zeros(n)
    for idx0 in tqdm(range(n)):
        sig_noise = np.fft.ifft2(add_noise_gaussian(sig_fft, sigma=noise_level))
        sig_cart = sig_noise[:, cart_x, cart_y]
        sig_musl = sig_noise[:, musl_x, musl_y]
        sig_synov = sig_noise[:, synov_x, synov_y]


        _Meff, T1map_cart[idx0], T2map_cart[idx0], _local_df = planet(
            sig_cart, alpha=alpha, TR=TR, T1_guess=2, pcs=pcs, compute_df=True)
        _Meff, T1map_musl[idx0], T2map_musl[idx0], _local_df = planet(
            sig_musl, alpha=alpha, TR=TR, T1_guess=2, pcs=pcs, compute_df=True)
        _Meff, T1map_synov[idx0], T2map_synov[idx0], _local_df = planet(
            sig_synov, alpha=alpha, TR=TR, T1_guess=2, pcs=pcs, compute_df=True)
        Meff_cart[idx0] = np.sum(np.abs(sig_cart))/npcs
        Meff_musl[idx0] = np.sum(np.abs(sig_musl))/npcs
        Meff_synov[idx0] = np.sum(np.abs(sig_synov))/npcs
    # number_of_bins=100
    #
    # plt.hist(T1map*1000, bins=number_of_bins)
    # plt.xlabel(phantom[x, y, 1]*1000)
    # plt.show()
    num_bins = 50

    fig0, axs0 = plt.subplots(3, 2, figsize=(12.8, 6.4 * 3))
    # fig1, axs1 = plt.subplots(1, 2, figsize=(12.8, 6.4))
    # fig2, axs2 = plt.subplots(1, 2, figsize=(12.8, 6.4))
    # the histogram of the data
    dict_result = {
        0: T1map_cart * 1000,
        1: T2map_cart * 1000,
        2: T1map_musl * 1000,
        3: T2map_musl * 1000,
        4: T1map_synov * 1000,
        5: T2map_synov * 1000
    }
    dict_x_label = {
        0: 'T1(ms) Nominal Value: %d ms' % (phantom[cart_x, cart_y, 1] * 1000),
        1: 'T2(ms) Nominal Value: %d ms' % (phantom[cart_x, cart_y, 2] * 1000),
        2: 'T1(ms) Nominal Value: %d ms' % (phantom[musl_x, musl_y, 1] * 1000),
        3: 'T2(ms) Nominal Value: %d ms' % (phantom[musl_x, musl_y, 2] * 1000),
        4: 'T1(ms) Nominal Value: %d ms' % (phantom[synov_x, synov_y, 1] * 1000),
        5: 'T2(ms) Nominal Value: %d ms' % (phantom[synov_x, synov_y, 2] * 1000)
    }
    dict_y_label = {
        0: 'Probability density'
    }
    dict_title = {
        0: 'T1 Distribution (Tissue: Cartilage)',
        1: 'T2 Distribution (Tissue: Cartilage)',
        2: 'T1 Distribution (Tissue: Muscle)',
        3: 'T2 Distribution (Tissue: Muscle)',
        4: 'T1 Distribution (Tissue: Synovial fluid)',
        5: 'T2 Distribution (Tissue: Synovial fluid)'
    }
    dict_alphabet = {
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f'
    }
    dict_snr={
    0:np.sum(Meff_cart)/(rayleigh_correction(noise_level)*n),
        1: np.sum(Meff_musl) / (rayleigh_correction(noise_level) * n),
        2: np.sum(Meff_synov) / (rayleigh_correction(noise_level) * n),
    }

    idx1, idx2 = 0, 0
    x_label_size = 15
    y_label_size = 10
    title_size = 15
    loc_text_upperR = .99
    loc_text_upperL = .01
    loc_Legend = (.75, .75)
    for idx00 in range(len(dict_result)):

        _n, bins, _patches = axs0[idx1, idx2].hist(dict_result.get(idx00, None), bins=num_bins, density=True,
                                                   stacked=True)
        (mu, sigma) = stats.norm.fit(dict_result.get(idx00, None))

        y = stats.norm.pdf(bins, mu, sigma)
        l = axs0[idx1, idx2].axvline(mu, color='r', linestyle='--', linewidth=2, label='Data_Mean')
        text_axs = 'SNR: %.4f\n' % dict_snr.get(idx1) + '$\\alpha=%.1f$' % np.rad2deg(
            alpha) + u'\N{DEGREE SIGN}' + '\n' + 'TR: %dms\n' % (
                           TR * 1000) + 'Number of Phase Cycles = %d\n' % npcs + 'Iteration No.: %d' % n + \
                   '\nData :$\mu=%.2f$, $\sigma=%.4f$' % (mu, sigma)
        axs0[idx1, idx2].set_xlabel(dict_x_label.get(idx00, None),
                                    fontdict={'fontsize': x_label_size})
        axs0[idx1, idx2].set_ylabel(dict_y_label.get(0, None), fontdict={'fontsize': y_label_size})
        axs0[idx1, idx2].set_title(dict_title.get(idx00, None), weight='bold', fontdict={'fontsize': title_size})
        axs0[idx1, idx2].legend(loc=loc_Legend)
        axs0[idx1, idx2].text(loc_text_upperR, loc_text_upperR, text_axs,
                              horizontalalignment='right',
                              verticalalignment='top',
                              transform=axs0[idx1, idx2].transAxes)
        axs0[idx1, idx2].text(loc_text_upperL, loc_text_upperR, dict_alphabet.get(idx00, None),
                              fontdict={'fontsize': 35}, weight='bold',
                              horizontalalignment='left',
                              verticalalignment='top',
                              transform=axs0[idx1, idx2].transAxes)
        # controlling loop
        idx2 += 1
        if idx2 == 2:
            idx2 = 0
            idx1 += 1

    plt.show()
    return


def showplot_single(plot, title='', cbar_title=''):
    cmap = 'viridis'
    plt.figure(1, figsize=(6.4, 4.8))
    plt.imshow(plot, origin='upper', interpolation=None, cmap=cmap)

    plt.title(title, fontdict={'fontsize': 20})
    clb = plt.colorbar()
    clb.ax.set_title(cbar_title)
    plt.show()
    return


def show_phantom(phantom):
    assert phantom.shape[2] == 5 or 6, 'Wrong dimension of input array'
    cmap = 'viridis'
    fig0, axs0 = plt.subplots(1, 6,figsize=(12.8, 12.8/5))

    dict_alphabet = {
        0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',

    }
    dict_title={
        0: 'M0',
        1: 'T1',
        2: 'T2',
        3: 'FA',
        4: 'Field Map',
        5: 'Raw data',
    }
    dict_ax_title={
        0: '',
        1: 'Sec',
        2: 'Sec',
        3: 'Rad',
        4: 'Hz',
        5: 'Arbitrary label',
    }

    fig0.subplots_adjust(top=1.2)

    for idx00 in range(6):
        axs0[idx00].set_title(dict_title.get(idx00, None), fontdict={'fontsize': 13})
        im = axs0[idx00].imshow(phantom[:,:,idx00],cmap = 'viridis')
        cbar = fig0.colorbar(im, ax=axs0[idx00])
        cbar.ax.set_title(dict_ax_title.get(idx00, None))
        axs0[idx00].text(.01, .99, dict_alphabet.get(idx00, None), fontdict={'fontsize': 25}, weight='bold',
                              horizontalalignment='left', color='w',
                              verticalalignment='top',
                              transform=axs0[idx00].transAxes)

    plt.show()
    return


def get_phantom(phantom):
    assert phantom.shape[2] == 6, 'Last axes has to be 6!!'

    M0, T1, T2, flip_angle, df, sample = phantom[:, :, 0], \
                                         phantom[:, :, 1], \
                                         phantom[:, :, 2], \
                                         phantom[:, :, 3], \
                                         phantom[:, :, 4], \
                                         phantom[:, :, 5]

    return M0, T1, T2, flip_angle, df, sample


if __name__ == '__main__':
    pass
