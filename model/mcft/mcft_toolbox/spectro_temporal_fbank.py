import numpy as np
#from scipy.fftpack import fft,ifft,fft2,ifft2,fftn, helper
from scipy.fftpack import helper


def gen_fbank_scale_rate(scale_ctrs,rate_ctrs,nfft_scale,nfft_rate,filt_params,comp_specgram=None,fbank_out_domain='sr'):
    """
    This function generates a scale-rate domain bank of up-/down-ward,filters.
    The filterbank will be tuned to the passband of a target signal if specified.
    Inputs:
    scale_ctrs: numpy array containing filter centers along the scale axis
    rate_ctrs: numpy array containing filter centers along the rate axis
    nfft_scale: number of scale fft points
    nfft_rate: number of rate fft points
    filt_params: dictionary containing filter parameters including:
                 samprate_spec: sample rate along the freq. axis (cyc/oct)
                 samprate_temp: sample rate along the time axis (cyc/sec)
                 time_const: exponent coefficient of the exponential term
    comp_spectgram: numpy array of size [num_freq_bin, num_time_frame] containing a complex spectrogram.
                    If provided, the function will return a filterbank that is modulated with the
                    phase of the spectrogram. Otherwise, the function will return the original set
                    of filters.
    fbank_out_domain: string indicating the representatio domain of the filter bank
                     'sr' (default): filter representation in the scale-rate domain
                     'tf': filter representation in the time-domain, involves an extra 2DIFT computation
                     'all': return filter represntations in both domains
    Output:
    fbank_2d_out: numpy array containing the 2D filter bank represented in the scale-rate or time-frequency domain
                 or list of length 2 containing filter representations in both domains
    Note: nfft_scale >= num_freq_bin and nfft_rate >= num_time_frame, where
          [num_freq_bin,num_time_frame] = np.shape(spectrogram)
    Note: the first and last filters in the scale and rate ranges are assumed
          to be lowpass and highpass respectively
    Author: Fatemeh Pishdadian (fpishdadian@u.northwestern.edu)
    """

    # check the 'comp_specgram' parameter
    if comp_specgram is None:
        mod_filter = 0
    else:
        mod_filter = 1

    ### Parameters and dimensions

    # set nfft to the next 5-smooth number
    nfft_scale = helper.next_fast_len(nfft_scale) #nfft_scale + np.mod(nfft_scale, 2)
    nfft_rate = helper.next_fast_len(nfft_rate) # nfft_rate + np.mod(nfft_rate, 2)

    num_scale_ctrs = len(scale_ctrs)
    num_rate_ctrs = len(rate_ctrs)

    beta = filt_params['time_const']
    samprate_spec = filt_params['samprate_spec']
    samprate_temp = filt_params['samprate_temp']

    scale_params = {'scale_filt_len': nfft_scale, 'samprate_spec': samprate_spec}
    rate_params = {'time_const': beta, 'rate_filt_len': nfft_rate, 'samprate_temp': samprate_temp}

    scale_filt_types = ['lowpass'] + (num_scale_ctrs-2) * ['bandpass'] + ['highpass']
    rate_filt_types = ['lowpass'] + (num_rate_ctrs-2) * ['bandpass'] + ['highpass']

    ### Generate the filterbank

    # filter modulation factor (pre-filtering stage)
    if mod_filter:
        # adjust the dimensions of the complex spectrogram
        comp_specgram = np.fft.ifft2(np.fft.fft2(comp_specgram, [nfft_scale, nfft_rate]))
        spec_phase = np.angle(comp_specgram)
        # uncomment this line to compare to the matlab code
        # otherwise fft phase difference results in large error values
        #spec_phase = np.angle(comp_specgram) * (np.abs(comp_specgram)>1e-10)
        filt_mod_factor = np.exp(1j * spec_phase)


    ### non-modulated filterbank:
    if not mod_filter:

        # return a single output domain
        if fbank_out_domain is not 'all': # 'tf' or 'sr'

            fbank_2d_out = np.zeros((num_scale_ctrs, 2 * num_rate_ctrs, nfft_scale, nfft_rate),dtype='complex128')

            for i in range(num_scale_ctrs): # iterate over scale filter centers
                scale_params['type'] = scale_filt_types[i]

                for j in range(num_rate_ctrs): # iterate over rate filter center
                    rate_params['type'] = rate_filt_types[j]

                    filt_up = gen_filt_scale_rate(scale_ctrs[i], rate_ctrs[j], scale_params, rate_params, 'up'
                                                     , filt_out_domain=fbank_out_domain)
                    filt_down = gen_filt_scale_rate(scale_ctrs[i], rate_ctrs[j], scale_params, rate_params, 'down'
                                                       , filt_out_domain=fbank_out_domain)

                    fbank_2d_out[i, num_rate_ctrs - j - 1, :, :] = filt_up
                    fbank_2d_out[i, num_rate_ctrs + j, :, :] = filt_down

        # return both output domains
        elif fbank_out_domain is 'all':

            fbank_tf_domain = np.zeros((num_scale_ctrs, 2 * num_rate_ctrs, nfft_scale, nfft_rate),dtype='complex128')
            fbank_sr_domain = np.zeros((num_scale_ctrs, 2 * num_rate_ctrs, nfft_scale, nfft_rate),dtype='complex128')

            for i in range(num_scale_ctrs): # iterate over scale filter centers
                scale_params['type'] = scale_filt_types[i]

                for j in range(num_rate_ctrs): # iterate over rate filter center
                    rate_params['type'] = rate_filt_types[j]

                    filt_up = gen_filt_scale_rate(scale_ctrs[i], rate_ctrs[j], scale_params, rate_params, 'up'
                                                  , filt_out_domain=fbank_out_domain)
                    filt_down = gen_filt_scale_rate(scale_ctrs[i], rate_ctrs[j], scale_params, rate_params, 'down'
                                                    , filt_out_domain=fbank_out_domain)

                    fbank_tf_domain[i, num_rate_ctrs - j - 1, :, :] = filt_up[0]
                    fbank_tf_domain[i, num_rate_ctrs + j, :, :] = filt_down[0]

                    fbank_sr_domain[i, num_rate_ctrs - j - 1, :, :] = filt_up[1]
                    fbank_sr_domain[i, num_rate_ctrs + j, :, :] = filt_down[1]

            fbank_2d_out = [fbank_tf_domain, fbank_sr_domain]



    ### modulated filterbank:
    if mod_filter:

        fbank_tf_domain = np.zeros((num_scale_ctrs, 2 * num_rate_ctrs, nfft_scale, nfft_rate), dtype='complex128')

        for i in range(num_scale_ctrs): # iterate over scale filter centers
            scale_params['type'] = scale_filt_types[i]

            for j in range(num_rate_ctrs): # iterate over rate filter center
                rate_params['type'] = rate_filt_types[j]

                # upward
                filt_tf_up = gen_filt_scale_rate(scale_ctrs[i], rate_ctrs[j], scale_params, rate_params,'up',
                                                 filt_out_domain='tf')
                filt_tf_up_mod = filt_tf_up * filt_mod_factor
                fbank_tf_domain[i, num_rate_ctrs - j - 1, :, :] = filt_tf_up_mod

                # downward
                filt_tf_down = gen_filt_scale_rate(scale_ctrs[i], rate_ctrs[j], scale_params, rate_params,'down',
                                                 filt_out_domain = 'tf')
                filt_tf_down_mod = filt_tf_down * filt_mod_factor
                fbank_tf_domain[i, num_rate_ctrs + j, :, :] = filt_tf_down_mod

        if fbank_out_domain is 'tf':
            fbank_2d_out = fbank_tf_domain

        elif fbank_out_domain is 'sr':
            fbank_2d_out = np.fft.fftn(fbank_tf_domain,axes=[2,3])

        elif fbank_out_domain is 'all':
            fbank_sr_domain = np.fft.fftn(fbank_tf_domain, axes=[2, 3])
            fbank_2d_out = [fbank_tf_domain, fbank_sr_domain]

    return fbank_2d_out



# MATLAB function: gen_hsr

def gen_filt_scale_rate(scale_ctr, rate_ctr, scale_params, rate_params, filt_dir,filt_out_domain='sr'):
    """
    This function generates a 2D-impulse response in the time-frequency
    domain with dilation factors S and R:
    The impulse response is denotes by h(omega, tau; S, R), where
    omega: frequency, tau: time,
    S: filter center on the scale axis, R: filter center on the rate axis
    Inputs:
    scale_ctr: filter center along the scale axis
    rate_ctr: filter center along the rate axis
    scale_params: dictionary containing the parameters of the spectral filter, including
                  scale_filt_len: length of the spectral filter impulse response
                  samprate_spec: sample rate along the freq. axis (cyc/oct)
                  type: string argument indicating the filter type
                        ('bandpass','lowpass','highpass')
    Example: scale_params = {'scale_filt_len':100,'samprate_spec':12,'type':'bandpass'}
    rate_params: dictionary containing the parameters of the temporal filter, including
                 time_const: exponent coefficient of the exponential term
                 rate_filt_len: length of the temporal filter impulse response
                 samprate_temp: sample rate along the time axis (cyc/sec)
                 type: string argument indicating the type of the filter
                      ('bandpass','lowpass','highpass')
    Example: rate_params = {'time_cons':1, rate_filt_len:200, samprate_temp:20, type:'lowpass'}
    filt_type: string type, determines the moving direction of the filter
              'none' (full s-r domain)
              'up' (upward analytic, nonzero over upper left (2nd) and lower right (4th) quadrants)
              'down' (downward analytic, nonzero over upper right (1st) and lower left (3rd) quadrants)
    filt_out_domain: string indicating the representatio domain of the filter
                     'sr' (default): filter representation in the scale-rate domain
                     'tf': filter representation in the time-domain, involves an extra 2DIFT computation
                     'all': return filter represntations in both domains
    Output:
    filt_2d_out: numpy array containing the 2D filter represented in scale-rate or time-frequency domain
                 or list of length 2 containing filter representations in both domains
    Author: Fatemeh Pishdadian (fpishdadian@u.northwestern.edu)
    """

    ### extract filter parameters

    # scale filter
    scale_filt_len = scale_params['scale_filt_len']
    samprate_spec = scale_params['samprate_spec']
    scale_filt_type = scale_params['type']

    # rate filter
    beta = rate_params['time_const']
    rate_filt_len = rate_params['rate_filt_len']
    samprate_temp = rate_params['samprate_temp']
    rate_filt_type = rate_params['type']

    ### frequency and time vectors

    # zero-pad filters to the next 5-smooth number
    scale_filt_len = helper.next_fast_len(scale_filt_len) # scale_filt_len + np.mod(scale_filt_len, 2)
    rate_filt_len = helper.next_fast_len(rate_filt_len) # rate_filt_len + np.mod(rate_filt_len, 2)

    # generate frequency and time vectors
    freq_vec = np.arange(scale_filt_len,dtype='float64')/samprate_spec
    time_vec = np.arange(rate_filt_len,dtype='float64')/samprate_temp


    ### impulse response of the original scale filter: Gaussian
    scale_filt = scale_ctr * (1 - 2 * (scale_ctr * np.pi * freq_vec)**2) * np.exp(-((scale_ctr * freq_vec * np.pi)**2))
    # make it even so the transform is real
    scale_filt = np.append(scale_filt[0:int(scale_filt_len/2)+1],scale_filt[int(scale_filt_len/2)-1:0:-1])

    ### impulse response of the original rate filter
    rate_filt = rate_ctr * (rate_ctr*time_vec)**2 * np.exp(-time_vec * beta * rate_ctr) * np.sin(2 * np.pi * rate_ctr * time_vec)
    # remove the DC element
    rate_filt = rate_filt - np.mean(rate_filt)
    # if the magnitude of dc element is set to zero by subtracting the mean of hr, make sure the phase is
    # also set to zero to avoid any computational error
    if np.abs(np.mean(rate_filt)) < 1e-16:
        correct_rate_phase = 1
    else:
        correct_rate_phase = 0

    ### scale response (Fourier transform of the scale impulse response)

    # bandpass scale filter
    scale_filt_fft = np.abs(np.fft.fft(scale_filt,n=scale_filt_len)).astype('complex128') # discard negligible imaginary parts

    # low/high-pass scale filter
    if scale_filt_type is not 'bandpass':
        scale_filt_fft_1 = scale_filt_fft[0:int(scale_filt_len/2)+1]
        scale_filt_fft_1 /= np.max(scale_filt_fft_1)
        max_idx_1 = np.squeeze(np.argwhere(scale_filt_fft_1 == np.max(scale_filt_fft_1)))

        scale_filt_fft_2 = scale_filt_fft[int(scale_filt_len/2)+1::]
        scale_filt_fft_2 /= np.max(scale_filt_fft_2)
        max_idx_2 = np.squeeze(np.argwhere(scale_filt_fft_2 == np.max(scale_filt_fft_2)))


        if scale_filt_type is 'lowpass':
            scale_filt_fft_1[0:max_idx_1] = 1
            scale_filt_fft_2[max_idx_2+1::] = 1

        elif scale_filt_type is 'highpass':
            scale_filt_fft_1[max_idx_1+1::] = 1
            scale_filt_fft_2[0:max_idx_2] = 1

        # form the full magnitude spectrum
        scale_filt_fft = np.append(scale_filt_fft_1, scale_filt_fft_2)



    ### rate response (Fourier transform of the rate impulse response)

    # band-pass rate filter
    rate_filt_fft = np.fft.fft(rate_filt, n=rate_filt_len) # rate response is complex

    # low/high-pass rate filter
    if rate_filt_type is not 'bandpass':
        rate_filt_phase = np.unwrap(np.angle(rate_filt_fft))
        if correct_rate_phase:
            rate_filt_phase[0] = 0
        rate_filt_mag = np.abs(rate_filt_fft)

        rate_filt_mag_1 = rate_filt_mag[0:int(rate_filt_len/2)+1]
        rate_filt_mag_1 /= np.max(rate_filt_mag_1)
        max_idx_1 = np.squeeze(np.argwhere(rate_filt_mag_1 == np.max(rate_filt_mag_1)))

        rate_filt_mag_2 = rate_filt_mag[int(rate_filt_len/2)+1::]
        rate_filt_mag_2 /= np.max(rate_filt_mag_2)
        max_idx_2 = np.squeeze(np.argwhere(rate_filt_mag_2 == np.max(rate_filt_mag_2)))

        if rate_filt_type is 'lowpass':
            rate_filt_mag_1[0:max_idx_1] = 1
            rate_filt_mag_2[max_idx_2+1::] = 1

        elif rate_filt_type is 'highpass':
            rate_filt_mag_1[max_idx_1+1::] = 1
            rate_filt_mag_2[0:max_idx_2+1] = 1

        # form the full magnitude spectrum
        rate_filt_mag = np.append(rate_filt_mag_1,rate_filt_mag_2)
        # form the full Fourier transform
        rate_filt_fft = rate_filt_mag * np.exp(1j * rate_filt_phase)


    ### full scale-rate impulse and transform responses

    # filt_sr_full is quadrant separable
    scale_filt_fft = np.expand_dims(scale_filt_fft,axis=1)
    rate_filt_fft = np.expand_dims(rate_filt_fft,axis=0)

    filt_sr_full = np.matmul(scale_filt_fft, rate_filt_fft)


    # normalize the filter magnitude
    filt_sr_full_mag = np.abs(filt_sr_full)
    filt_sr_full_mag /= np.max(filt_sr_full_mag)

    filt_sr_full_phase = np.angle(filt_sr_full)
    filt_sr_full = filt_sr_full_mag * np.exp(1j * filt_sr_full_phase)


    # upward or downward direction
    if filt_dir is 'up':
        # compute the upward version of the scale-rate response
        filt_sr_up = filt_sr_full
        filt_sr_up[1:int(scale_filt_len/2)+1, 1:int(rate_filt_len/2)+1] = 0
        filt_sr_up[int(scale_filt_len/2)+1::,int(rate_filt_len/2)+1::] = 0
        filt_sr_domain = filt_sr_up

    elif filt_dir is 'down':
        # compute the downward version of the scale-rate response
        filt_sr_down = filt_sr_full
        filt_sr_down[1:int(scale_filt_len/2)+1,int(rate_filt_len/2)+1::] = 0
        filt_sr_down[int(scale_filt_len/2)+1::,1:int(rate_filt_len/2)+1] = 0
        filt_sr_domain = filt_sr_down

    else:
        filt_sr_domain = filt_sr_full


    if filt_out_domain is 'sr':
        filt_out = filt_sr_domain
    else:
        filt_tf_domain = np.fft.ifft2(filt_sr_domain)

        if np.max(np.imag(filt_tf_domain)) < 1e-8:
            filt_tf_domain = np.real(filt_tf_domain)

        filt_out = filt_tf_domain

        if filt_out_domain is 'all':
            filt_out = [filt_out, filt_sr_domain]

    return filt_out


# ToDo add Smin, Smax, Rmin and Rmax to parameters and an option for using them instead of automatically
# selecting min and max values

def filt_default_centers(scale_params,rate_params):
    """
    This function computes the default set of 2D filter centers along scale and rate axes.
    Inputs:
    scale_params: tuple containing parameters of scale filters, including:
                  scale_res: number of bins per octave on the scale axis
                  scale_nfft: number of fft points on the scale axis
                  samprate_spec: sampling rate of the spectral filter (in cycles per octave)
                                 scale_max = samprate_spec/2
    rate_params: tuple containing parameters of rate filters, including:
                 rate_res: number of bins per octave on the rate axis
                 rate_nfft: number of fft points on the rate axis
                 samprate_temp: sampling rate of the temporal filters (in cycles per second)
    Outputs:
    scale_ctrs: numpy array containing scale filter centers
    rate_ctrs: numpy array containing rate filter centers
    Author: Fatemeh Pishdadian (fpishdadian@u.northwestern.edu)
    """

    # extract scale and rate parameters
    scale_res, scale_nfft, samprate_spec = scale_params
    rate_res, rate_nfft, samprate_temp = rate_params

    # compute scale filter centers
    scale_ctrs = filt_centers('scale',scale_res, scale_nfft, samprate_spec)

    # compute rate filter centers
    rate_ctrs = filt_centers('rate',rate_res, rate_nfft, samprate_temp)

    return scale_ctrs, rate_ctrs



def filt_centers(filt_type,bins_per_oct,nfft,samprate):
    """
    This function computes scale filter centers given transform-domain parameters.
    Inputs:
    filt_type: string, 'scale' or 'rate' (high pass filter is computed differently based on filter type)
    bins_per_oct: number of scale or rate filters per octave
    nfft: number of frequencies of analysis in the scale domain
    samprate: sampling rate in the spectral domain
    Outputs:
    scale_ctrs: numpy array containing filter centers
    """

    # spacing between frequencies of analysis
    grid_res = samprate/nfft

    # center of the lowpass filter
    ctr_low = grid_res/2

    # center of the highest bandpass filter
    log2_ctr_band_min = np.ceil(np.log2(ctr_low))
    # add 1 to power of 2 if the smallest bandpass center is smaller the lowpass center
    log2_ctr_band_min += float(2**log2_ctr_band_min <= ctr_low)
    log2_ctr_band_max = np.floor(np.log2(samprate/2))


    # centers of bandpass filters
    ctr_band = 2 ** np.arange(log2_ctr_band_min, log2_ctr_band_max+(1/bins_per_oct), 1/bins_per_oct)

    # center of the highpass filter
    if filt_type is 'scale':
        ctr_high = (ctr_band[-1] + 3 * samprate / 2) / 4
    elif filt_type is 'rate':
        ctr_high = (ctr_band[-1] + 1 * samprate / 2) / 2

    # shift filter centers to the nearest frequency of analysis
    ctr_band = np.round(ctr_band/grid_res) * grid_res
    ctr_high = np.round(ctr_high/grid_res) * grid_res

    # concatenate all centers into one vector
    ctr_all = np.append(ctr_low, ctr_band)
    ctr_all = np.append(ctr_all,ctr_high)

    # remove repeated values(sometimes happens due to rounding)
    filt_ctrs = np.unique(ctr_all)

    return filt_ctrs
