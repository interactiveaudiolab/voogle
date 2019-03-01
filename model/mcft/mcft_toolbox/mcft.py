import numpy as np
from scipy.fftpack import fft2,ifft2,fftn,ifftn, fft, ifft
from model.mcft.cqt_toolbox.cqt import cqt
from model.mcft.mcft_toolbox.spectro_temporal_fbank import filt_default_centers,gen_fbank_scale_rate


def mcft(signal, cqt_params_in, filt_params_in=None,del_cqt_phase=0):
    """
    This function applies the Multi-resolution Common Fate
    Transform (MCFT) to a time-domain audio signal.
    The intermediary time-frequency domain representation is the
    Constant-Q Transform (CQT) of the audio signal, which is
    computed using the invertible and optimized CQT implementation
    proposed by Schorkhuber et al.:
    Toolbox webpage:
    http://www.cs.tut.fi/sgn/arg/CQT/
    Reference:
    Schörkhuber et al. "A Matlab toolbox for efficient perfect reconstruction
    time-frequency transforms with log-frequency resolution."
    Audio Engineering Society Conference: 53rd International Conference:
    Semantic Audio. Audio Engineering Society, 2014.
    The python translation of this CQT toolbox can be found here:
    https://github.com/interactiveaudiolab/MCFT/tree/master/mcft_python/cqt_toolbox
    Inputs:
    signal: numpy array containing samples of the time-domain signal
    cqt_params_in: dictionary contatining CQT parameters, including:
                   samprate_sig: sampling rate of the audio signal
                   fmin: center frequency of the lowest cqt filter
                   fmax: center frequency of the highest cqt filter
                   fres: frequency resolution (number of bins per octave)
                   gamma: linear-Q factor
    filt_params_in: dictionary containing parameters of of 2d (spectro-temporal)
                   filters, including:
                   scale_ctrs: numpy array containing scale filter centers
                   rate_ctrs: numpy array containing rate filter centers
                   time_const: time constant of the temporal filter
    del_cqt_phase: boolean indicating whether the cqt phase is to be deleted or
                   included in the 2d filtering process.
                   If the phase is included, the filterbank will be modulated with the phase.
                   If the phase is deleted the origianl filterbank will be applied to the
                   magnitude CQT.
                   Note: the output representation is invertible only if the phase is included (default)
    Outputs:
    mcft_out: 4d numpy array containing the MCFT coefficients
    cqt_params_out: dictionary containing all CQT parameters (required for reconstruction)
    fbank_sr_domain: 4d numpy array containing the scale-rate filterbank
    Author: Fatemeh Pishdadian (fpishdadian@u.northwestern.edu)
    """

    # check filter parameters
    if filt_params_in is None:
        set_filt_params = 1
    else:
        set_filt_params = 0
        scale_ctrs = filt_params_in['scale_ctrs']
        rate_ctrs = filt_params_in['rate_ctrs']
        beta = filt_params_in['time_const']


    # extract cqt parameters
    samprate_sig = cqt_params_in['samprate_sig']
    fmin = cqt_params_in['fmin']
    fmax = cqt_params_in['fmax']
    fres = cqt_params_in['fres']
    gamma = cqt_params_in['gamma']

    ### compute the CQT of the time-domain signal

    # cqt transform
    cqt_results = cqt(signal, fres, samprate_sig, fmin, fmax, gamma=gamma)
    sig_cqt = cqt_results['cqt']

    num_freq_bin, num_time_frame = np.shape(sig_cqt)

    # cqt output parameters
    cqt_params_out = cqt_results
    del cqt_params_out['cqt']
    cqt_params_out['num_freq_bin'] = num_freq_bin
    cqt_params_out['num_time_frame'] = num_time_frame

    ### parameters of the spectro-temporal filters

    sig_dur = len(signal) / samprate_sig # signal duration in seconds
    nfft_scale = num_freq_bin # mininum number of fft points on the scale axis
    nfft_rate = num_time_frame # minimum number of fft points on the rate axis
    samprate_spec = fres # sampling rate of the spectral filters (in cyc/oct)
    samprate_temp = np.floor(num_time_frame / sig_dur) # sampling rate of the temporal filters (in cyc/sec)

    # compute scale and rate filter centers if not provided
    if set_filt_params:

        beta = 1 # time constant of the tempral filter

        scale_res = 1 # number of scale filters per octave
        rate_res = 8 # number of rate filters per octave
        scale_params = (scale_res,nfft_scale,samprate_spec)
        rate_params = (rate_res, nfft_rate, samprate_temp)

        scale_ctrs, rate_ctrs = filt_default_centers(scale_params,rate_params)

    # concatenate filter parameters into one dictionary
    filt_params_dict = {'samprate_spec':samprate_spec, 'samprate_temp':samprate_temp, 'time_const':beta}

    ### compute the filterbank
    print('Computing the filterbank ...')

    if del_cqt_phase:
        _, fbank_sr_domain = gen_fbank_scale_rate(scale_ctrs, rate_ctrs, nfft_scale, nfft_rate, filt_params_dict)
    else:
        _, fbank_sr_domain = gen_fbank_scale_rate(scale_ctrs, rate_ctrs, nfft_scale, nfft_rate, filt_params_dict,
                                                  comp_specgram=sig_cqt)

    ### compute the MCFT
    print('Computing the transform...')

    if del_cqt_phase:
        sig_cqt = np.abs(sig_cqt)

    mcft_out = cqt_to_mcft(sig_cqt,fbank_sr_domain)


    return mcft_out, cqt_params_out, fbank_sr_domain


def cqt_to_mcft(sig_cqt,fbank_scale_rate,mcft_out_domain='tf'):
    """
    This function receives the time-frequency representation (CQT)
    of an audio signal (complex in general) and generates a 4-dimensional
    representation (scale,rate,frequency,time) by 2d filtering based
    on the cortical part of Chi's auditory model.
    Inputs:
    sig_cqt: 2d numpy array containing the (complex) time-frequency
             representation of an audio signal (log scale frequency, e.g. CQT)
    fbank_scale_rate: 4d numpy array containing a bank of filters in the
             scale-rate domain
    mcft_out_domain: string indicating whether the filtered specgrogram is returned
                     in the time-frequency domain or scale_rate domain
                     'tf' (default): time-frequency domain, involves an extra 2D-IFT step
                     'sr': scale-rate domain
    Ouptput:
    mcft_out: 4d numpy array containing the MCFT coefficients
    Author: Fatemeh Pishdadian (fpishdadian@u.northwestern.edu)
    """

    # dimensions
    num_scale_ctrs, num_rate_ctrs, nfft_scale, nfft_rate = np.shape(fbank_scale_rate)

    ### compute the MCFT coefficients

    # 2D-Fourier transform of the time-frequency representation
    sig_cqt_2dft = np.fft.fft(sig_cqt,nfft_scale,axis=0)
    sig_cqt_2dft = np.fft.fft(sig_cqt_2dft.T,nfft_rate,axis=0).T

    # allocate memory for the coefficients
    mcft_out = np.zeros((num_scale_ctrs, num_rate_ctrs, nfft_scale, nfft_rate), dtype='complex128')

    for i in range(num_scale_ctrs):
        for j in range(num_rate_ctrs):
            # extract the current filter
            filt_sr_temp = fbank_scale_rate[i, j, :, :]

            # filter the signal in the scale-rate domain
            sig_filt_sr = sig_cqt_2dft * filt_sr_temp

            # convert back to the time-frequency domain if specified
            if mcft_out_domain is 'tf':
                sig_filt_tf = np.fft.ifft(sig_filt_sr,axis=0)
                sig_filt_tf = np.fft.ifft(sig_filt_tf.T,axis=0)
                mcft_out[i, j, :, :] = sig_filt_tf.T
            elif mcft_out_domain is 'sr':
                mcft_out[i, j, :, :] = sig_filt_sr

    return mcft_out



# def cqt_to_mcft(sig_cqt,fbank_scale_rate,mcft_out_domain='tf'):
#     """
#     This function receives the time-frequency representation (CQT)
#     of an audio signal (complex in general) and generates a 4-dimensional
#     representation (scale,rate,frequency,time) by 2d filtering based
#     on the cortical part of Chi's auditory model.
#
#     Inputs:
#     sig_cqt: 2d numpy array containing the (complex) time-frequency
#              representation of an audio signal (log scale frequency, e.g. CQT)
#     fbank_scale_rate: 4d numpy array containing a bank of filters in the
#              scale-rate domain
#
#     mcft_out_domain: string indicating whether the filtered specgrogram is returned
#                      in the time-frequency domain or scale_rate domain
#                      'tf' (default): time-frequency domain, involves an extra 2D-IFT step
#                      'sr': scale-rate domain
#
#     Ouptput:
#     mcft_out: 4d numpy array containing the MCFT coefficients
#
#     Author: Fatemeh Pishdadian (fpishdadian@u.northwestern.edu)
#     """
#
#     # dimensions
#     num_scale_ctrs, num_rate_ctrs, nfft_scale, nfft_rate = np.shape(fbank_scale_rate)
#
#     ### compute the MCFT coefficients
#
#     # 2D-Fourier transform of the time-frequency representation
#     sig_cqt_2dft = fft2(sig_cqt,[nfft_scale, nfft_rate])
#
#     # allocate memory for the coefficients
#     mcft_out = np.zeros((num_scale_ctrs, num_rate_ctrs, nfft_scale, nfft_rate), dtype='complex128')
#
#     for i in range(num_scale_ctrs):
#         for j in range(num_rate_ctrs):
#
#             # extract the current filter
#             filt_sr_temp = fbank_scale_rate[i,j,:,:]
#
#             # filter the signal in the scale-rate domain
#             sig_filt_sr = sig_cqt_2dft * filt_sr_temp
#
#             # convert back to the time-frequency domain if specified
#             if mcft_out_domain is 'tf':
#                 sig_filt_tf = ifft2(sig_filt_sr)
#                 mcft_out[i, j, :, :] = sig_filt_tf
#             elif mcft_out_domain is 'sr':
#                 mcft_out[i, j, :, :] = sig_filt_sr
#
#             # convert back to the time-frequency domain
#
#     return mcft_out
