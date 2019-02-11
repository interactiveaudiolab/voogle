from __future__ import print_function, division

import numpy as np

from model.mcft.cqt_toolbox.gen_filterbank import gen_filterbank
from model.mcft.cqt_toolbox.apply_filterbank import apply_filterbank

def cqt(signal, bins_per_octave, samp_rate, fmin, fmax,
            rasterize='full', phasemode='global', outputFormat='sparse',
            gamma=0, normalize='sine', window_name='hann'):
    # type: (numpy.ndarray, int, int, float, float, str, str, str, int, str, str) -> dict
    '''
    Input parameters:
          signal            : Real-valued signal
          bins_per_octave   : Number of bins per octave
          samp_rate         : Sampling frequency
          fmin              : Lowest frequency to be analyzed
          fmax              : Highest frequency to be analyzed
          **rasterize       : Can be none, full, or piecewise --
                              affects the hop size
          **phasemode       : Can be local or global -- global
                              uses a mapping function
          **outputFormat    : Can be sparse or cell to determine
                              datatype of returned coefficients
          **gamma           : Constant factor for offsetting filter bandwidths, >=0
          **normalize       : Can be sine, impulse or none -- used to
                              normalize the coefficients
          **window_name     : Name of the window function used to generate filter

    Output parameters:
          results          : Dict consisting of
             .cqt            : CQT coefficients
             .cqt_DC         : transform coefficients for f = 0
             .cqt_Nyq        : transform coefficients for nyquist rate
             .filter_bank    : list of analysis filters
             .shift          : center frequencies of analysis filters
             .bw_bins        : bandwidth of analysis filters
             .sig_len        : length of input signal
             .phasemode      : 'local'  -> zero-centered filtered used
                             : 'global' -> mapping function used
             .rast           : time-frequency plane sampling scheme (full,
                               piecewise, none)
             .fmin           : Lowest frequency to be analyzed
             .fmax           : Highest frequency to be analyzed
             .bins_er_octave : Number of bins per octave
             .format         : eihter 'cell' or 'matrix' (only applies for
                               piecewise rasterization)

    **optional args

    This function is a composition of gen_filter, gen_filterbank and apply_filterbank.
    It takes in or generates all of the necessary parameters for those functions
    and passes their output around to in order to compute the actual CQT. It requires
    a real-valued signal, a number of bins per octave, and a min and max frequency. In
    addition it allows you to tune values for the other functions like the window used
    to shape the filters, the gamma factor, and the phasemode.

    References:
      C. Schorkhuber, A. Klapuri, N. Holighaus, and M. Dorfler. A Matlab
      Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy
      Transforms.

      G. A. Velasco, N. Holighaus, M. Dorfler, and T. Grill. Constructing an
      invertible constant-Q transform with non-stationary Gabor frames.
      Proceedings of DAFX11, Paris, 2011.

      N. Holighaus, M. Dorfler, G. Velasco, and T. Grill. A framework for
      invertible, real-time constant-q transforms. Audio, Speech, and
      Language Processing, IEEE Transactions on, 21(4):775-785, April 2013.

    See also:  nsgtf_real, winfuns

    Translation from MATLAB by: Trent Cwiok (cwiok@u.northwestern.edu)
                                Fatemeh Pishdadian (fpishdadian@u.northwestern.edu)
    '''
    filter_bank,shift,bw_bins = gen_filterbank(fmin,fmax,bins_per_octave,samp_rate,len(signal), window_name=window_name, gamma=gamma)

    num_filters = int(len(bw_bins)/2 -1)
    ctr_freqs = samp_rate * np.cumsum(shift[1:]) / len(signal)
    ctr_freqs = ctr_freqs[:num_filters]

    # Assumes rasterize is full always
    bw_bins[1:num_filters+1] = bw_bins[num_filters]
    bw_bins[num_filters+2:] = bw_bins[num_filters:0:-1]

    # Create a normalization vector
    normalize = normalize.lower()
    if normalize in ['sine','sin']:
        normFacVec = 2 * bw_bins[:num_filters+2]/len(signal)
    elif normalize in ['impulse','imp']:
        filter_lens = np.zeros(len(filter_bank))
        for i in range(len(g)):
            filter_lens[i] = len(filter_bank[i])
        normFacVec = 2 * bw_bins[:num_filters+2]/filter_lens
    elif normalize in ['none','no']:
        normFacVec = np.ones(num_filters+2)
    else:
        print("Unknown normalization method")

    normFacVec = np.concatenate((normFacVec,normFacVec[len(normFacVec)-2:0:-1]))

    # Apply normalization to the filterbank
    for i in range(len(normFacVec)):
        filter_bank[i] *= normFacVec[i]

    if len(signal.shape) < 2:
        signal.shape = (len(signal),1)

    # Apply the normalized filterbank to the signal to compute the cqt
    cqt,sig_len = apply_filterbank(signal,filter_bank,shift,phasemode,bw_bins)

    # Assume rasterize is full always
    # Seperate  the actual cqt from the zero and nyq coefficients needed for perfect reconstruction
    cqt_DC = np.squeeze(cqt[0])
    cqt_Nyq = np.squeeze(cqt[num_filters+1])
    cqt = np.squeeze(np.asarray(cqt[1:num_filters+1]))

    results = {'cqt':cqt,'filter_bank':filter_bank,'shift':shift,'bw_bins':bw_bins,'sig_len':sig_len,
        'phasemode':phasemode,'rast':rasterize,'fmin':fmin,'fmax':fmax,
        'bins_per_octave':bins_per_octave,'cqt_DC':cqt_DC,'cqt_Nyq':cqt_Nyq,'format':outputFormat,'ctr_freqs':ctr_freqs}

    return results
