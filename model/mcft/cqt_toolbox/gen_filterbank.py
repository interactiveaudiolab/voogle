from __future__ import print_function, division

import numpy as np
from model.mcft.cqt_toolbox.gen_filter import gen_filter as gfilt

def gen_filterbank(fmin,fmax,bins_per_octave,samp_rate,sig_len,min_filt_len=4,bw_factor=1,fractional=False,window_name='hann',gamma=0):
    # type: (float, float, int, int, int, int, int, bool, str, int) -> (list, numpy.ndarray, numpy.ndarray)
    '''
    Input parameters:
        fmin                 : Minimum frequency (in Hz)
        fmax                 : Maximum frequency (in Hz)
        bins_per_octave      : number of bins_per_octave per octave
        samp_rate            : Sampling rate (in Hz)
        sig_len              : Length of signal (in samples)
        **min_filt_len       : Minimal filter length allowed
        **bw_factor          : Values in bw_bins are rounded to factors of this
        **fractional         : True if shifts can have fractional values
        **window_name        : Name of the window function used to generate filter
        **gamma              : Constant factor for offsetting filter bandwidths, >=0

    Output parameters:
        filter_bank         : List of ndarrays, each representing a different filter
        shift               : Ndarray of shifts between the center frequencies
        bw_bins             : Ndarray of bandwidths of the bins

    **optional args

    Create a nonstationary Gabor filterbank with constant or varying
    Q-factor and relevant frequency range from fmin to fmax. To allow
    for perfect reconstruction, the frequencies outside that range will be
    captured by 2 additional filters placed on the zero and Nyquist
    frequencies, respectively.

    The Q-factor (quality factor) is the ratio of center frequency to
    bandwidth cent_freq/bandwidth. The Q-factor is determined only by the
    bins_per_octave and the value of gamma. A gamma value of 0 implies
    a constant-Q filter. For gamma greater than 0, variable-Q filters are
    returned, which allow for better time-resolution in lower frequencies.

    For more details on the construction of the constant-Q nonstationary
    Gabor filterbank, please see the references.

    References:
        C. Schorkhuber, A. Klapuri, N. Holighaus, and M. Dorfler. A Matlab
        Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy
        Transforms.

        G. A. Velasco, N. Holighaus, M. DAJrfler, and T. Grill. Constructing an
        invertible constant-Q transform with non-stationary Gabor frames.
        Proceedings of DAFX11, Paris, 2011.

        N. Holighaus, M. DAJrfler, G. Velasco, and T. Grill. A framework for
        invertible, real-time constant-q transforms. Audio, Speech, and
        Language Processing, IEEE Transactions on, 21(4):775-785, April 2013.

    See also: nsgtf_real, winfuns

    Translation from MATLAB by: Trent Cwiok (cwiok@u.northwestern.edu)
                                Fatemeh Pishdadian (fpishdadian@u.northwestern.edu)
    '''
    # Calculate nyquist rate and redefine fmax in terms of nyquist
    nyquist = samp_rate/2
    if fmax > nyquist:
        fmax = nyquist

    #  Freq resolution, center frequency vector, and bandwidths in term of center freqs
    fftres = samp_rate/sig_len
    bins_total = np.floor(bins_per_octave * np.log2(fmax/fmin))
    ctr_freqs = fmin * 2**(np.asarray(np.arange(bins_total+1))/bins_per_octave)
    Q = 2**(1/bins_per_octave) - 2**(-1/bins_per_octave)
    cqt_bw = Q*ctr_freqs + gamma

    # Excludes frequencies above the nyquist rate
    nonzeroIndices = np.nonzero([int(ctr_freqs[i]+cqt_bw[i]/2>nyquist) for i in range(len(ctr_freqs))])
    if nonzeroIndices[0].size > 0:
        ctr_freqs = ctr_freqs[:nonzeroIndices[0][0]]
        cqt_bw = cqt_bw[:nonzeroIndices[0][0]]

    # Excludes frequencies below zero
    nonzeroIndices = np.nonzero([int(ctr_freqs[i]-cqt_bw[i]/2<0) for i in range(len(ctr_freqs))])
    if nonzeroIndices[0].size > 0:
        ctr_freqs = ctr_freqs[nonzeroIndices[0][-1]:]
        cqt_bw = cqt_bw[nonzeroIndices[0][-1]:]
        print("fmin set to ", None, " Hz!")

    # Get the number of filters needed, include zero and nyquist filters then mirror
    # [0,pi] for (pi,2pi). Also, convert bw and ctr_freqs from Hz to # of bins
    num_ctr_freqs = len(ctr_freqs)
    ctr_freqs = np.concatenate(([0],ctr_freqs,[nyquist],samp_rate-np.flip(ctr_freqs,0)))
    bw = np.concatenate(([2*fmin],cqt_bw,[ctr_freqs[num_ctr_freqs+2]-ctr_freqs[num_ctr_freqs]],np.flip(cqt_bw,0)))
    bw /= fftres
    ctr_freqs /= fftres

    # center positions of filters in DFT frame -- only whole number part
    ctr_freqs_int = np.zeros(ctr_freqs.size)
    ctr_freqs_int[:num_ctr_freqs+2] = np.floor(ctr_freqs[:num_ctr_freqs+2])
    ctr_freqs_int[num_ctr_freqs+2:] = np.ceil(ctr_freqs[num_ctr_freqs+2:])

    # Distance between center frequencies ofthe filters
    shift = np.concatenate(([-1*ctr_freqs_int[-1] % sig_len], np.diff(ctr_freqs_int)))

    # If we allow for fractional distances, calculate the fractional portion
    if fractional:
        frac_shift = ctr_freqs-ctr_freqs_int
        bw_bins = np.ceil(bw+1)
    else:
        bw = np.round(bw)
        bw_bins = bw

    # Set minimum bandwidth
    for i in range(2*(num_ctr_freqs+1)):
        if bw[i] < min_filt_len:
            bw[i] = min_filt_len
            bw_bins[i] = bw[i]

    # Generate the actual filterbank using the gen_filter function
    if fractional:
        filter_bank = []
        for i in range(len(bw_bins)):
            samples = np.concatenate((range(int(np.ceil(bw_bins[i]/2)+1)),range(int(-1*np.ceil(bw_bins[i]/2)),0)))
            samples -= frac_shift[i]
            samples /= bw[i]
            win = gfilt(window_name, sample_positions=samples)
            win /= np.sqrt(bw[i])
            filter_bank.append(win)
    else:
        filter_bank = []
        for i in range(len(bw)):
            filter_bank.append(gfilt(window_name,num_samples=bw[i]))

    # Round to multiples of the bw factor input arg
    bw_bins = bw_factor*np.ceil(bw_bins/bw_factor)

    # Apply a Tukey window at 0 and Nyquist to allow for perfect reconstruction
    for i in [0,num_ctr_freqs+1]:
        if bw_bins[i] > bw_bins[i+1]:
            filter_bank[i] = np.ones(int(bw_bins[i]))
            start = int(np.floor(bw_bins[i]/2)-np.floor(bw_bins[i+1]/2))
            end = int(np.floor(bw_bins[i]/2)+np.ceil(bw_bins[i+1]/2))
            filter_bank[i][start:end] = gfilt('hann',num_samples=bw_bins[i+1])
            filter_bank[i] /= np.sqrt(bw_bins[i])

    return filter_bank, shift, bw_bins
