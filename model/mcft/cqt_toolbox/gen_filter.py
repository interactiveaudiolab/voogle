from __future__ import print_function, division

import numpy as np

def gen_filter(window_name, sample_positions=None, num_samples=None):
    # type: (str, numpy.ndarray, float) -> numpy.ndarray
    '''
    Input parameters: 
          window_name          : String containing the window name
          **sample_positions   : Ndarray of sampling positions
          **num_samples        : Number of samples in the window
          
    Output parameters:
          g                    : Ndarray filter in specified window shape

    **optional args
    
    This function is used to generate an individual filter in the shape of the
    specified window. The filter can either be sampled over the specified 
    vector of points if sample_positions is NOT None. Before returning the
    filter it is masked to force everything outside of -.5 and .5 to zero. If 
    sample_positions is None, then num_samples must be provided. In this case, 
    the filter returned ranges from -.5 to .5 with num_samples of points. 

    The following windows are available: 

    'hann'         von Hann window. Forms a PU. The Hann window has a
                 mainlobe with of 8/N, a PSL of -31.5 dB and decay rate
                 of 18 dB/Octave. 

    'cos'          Cosine window. This is the square root of the Hanning
                 window. The cosine window has a mainlobe width of 6/N,
                 a  PSL of -22.3 dB and decay rate of 12 dB/Octave.
               
    'rec'          Rectangular window. The rectangular window has a
                 mainlobe width of 4/N, a  PSL of -13.3 dB and decay
                 rate of 6 dB/Octave. Forms a PU. Alias: 'square' 

    'tri'          Triangular window.  

    'hamming'      Hamming window. Forms a PU that sums to 1.08 instead
                 of 1.0 as usual. The Hamming window has a
                 mainlobe width of 8/N, a  PSL of -42.7 dB and decay
                 rate of 6 dB/Octave. 

    'blackman'     Blackman window. The Blackman window has a
                 mainlobe width of 12/N, a PSL of -58.1 dB and decay
                 rate of 18 dB/Octave. 

    'blackharr'    Blackman-Harris window. The Blackman-Harris window has 
                 a mainlobe width of 16/N, a PSL of -92.04 dB and decay
                 rate of 6 dB/Octave. 

    'modblackharr'  Modified Blackman-Harris window. This slightly 
                  modified version of the Blackman-Harris window has 
                  a mainlobe width of 16/N, a PSL of -90.24 dB and decay
                  rate of 18 dB/Octave. 

    'nuttall'      Nuttall window. The Nuttall window has a mainlobe 
                 width of 16/N, a PSL of -93.32 dB and decay rate of 
                 18 dB/Octave. 

    'nuttall10'    2-term Nuttall window with 1 continuous derivative. 
                 Alias: 'hann'. 

    'nuttall01'    2-term Nuttall window with 0 continuous derivatives. 
                 Alias: 'hamming'. 

    'nuttall20'    3-term Nuttall window with 3 continuous derivatives. 
                 The window has a mainlobe width of 12/N, a PSL of 
                 -46.74 dB and decay rate of 30 dB/Octave. 

    'nuttall11'    3-term Nuttall window with 1 continuous derivative. 
                 The window has a mainlobe width of 12/N, a PSL of 
                 -64.19 dB and decay rate of 18 dB/Octave. 

    'nuttall02'    3-term Nuttall window with 0 continuous derivatives. 
                 The window has a mainlobe width of 12/N, a PSL of 
                 -71.48 dB and decay rate of 6 dB/Octave. 

    'nuttall30'    4-term Nuttall window with 5 continuous derivatives. 
                 The window has a mainlobe width of 16/N, a PSL of 
                 -60.95 dB and decay rate of 42 dB/Octave. 

    'nuttall21'    4-term Nuttall window with 3 continuous derivatives. 
                 The window has a mainlobe width of 16/N, a PSL of 
                 -82.60 dB and decay rate of 30 dB/Octave. 

    'nuttall12'    4-term Nuttall window with 1 continuous derivatives. 
                 Alias: 'nuttall'. 

    'nuttall03'    4-term Nuttall window with 0 continuous derivatives. 
                 The window has a mainlobe width of 16/N, a PSL of 
                 -98.17 dB and decay rate of 6 dB/Octave. 

    'gauss'        Truncated, stretched Gaussian: exp(-18*x^2) restricted
                 to the interval ]-.5,.5[. 

    'wp2inp'       Warped Wavelet uncertainty equalizer (see WP 2 of the
                 EU funded project UnlocX). This function is included 
                 as a test function for the Wavelet transform 
                 implementation and serves no other purpose in this 
                 toolbox. 

    References:
      Wikipedia. Window function - wikipedia article.
      http://en.wikipedia.org/wiki/Window_function.
      
      A. Nuttall. Some windows with very good sidelobe behavior. IEEE Trans.
      Acoust. Speech Signal Process., 29(1):84-91, 1981.
      
      F. Harris. On the use of windows for harmonic analysis with the
      discrete Fourier transform. Proceedings of the IEEE, 66(1):51 - 83,
      January 1978.

    See also:  gen_filterbank, gen_inv_filterbank

    Translation from MATLAB by: Trent Cwiok (cwiok@u.northwestern.edu)
                                Fatemeh Pishdadian (fpishdadian@u.northwestern.edu)
    '''

    # Input argument checking -- either sample_positions or num_samples must be provided
    if sample_positions is not None:
        pass
    elif num_samples != None:
        step = 1/num_samples
        # Two cases if the num of samples is even or odd
        if num_samples % 2 == 0:
            # For even N the sampling interval is [0,.5-1/N] + [-.5,0)
            first_half = np.linspace(0,.5-step,int(num_samples*.5))
            second_half = np.linspace(-.5,-step,int(num_samples*.5))
            sample_positions = np.concatenate((first_half,second_half))
        else: 
            # For odd N the sampling interval is [0,.5-1/(2N)] + [-.5+1/(2N),0) 
            first_half = np.linspace(0,.5-.5*step,int(num_samples*.5)+1)
            second_half = np.linspace(-.5+.5*step,-step,int(num_samples*.5))
            sample_positions = np.concatenate((first_half,second_half))
    else:
        print("Error: invalid arguements to window function generator.")
        return None


    # Switch case for possible window names, forcing everything to lowercase
    window_name = window_name.lower()
    if window_name in ['hann','nuttall10']:
        filter_ = .5 + .5*np.cos(2*np.pi*sample_positions)
        
    elif window_name in ['cosine','cos','sqrthann']:
        filter_ = np.cos(np.pi*sample_positions)
        
    elif window_name in ['hamming','nuttall01']:
        filter_ = .54 + .46*np.cos(2*np.pi*sample_positions)
        
    elif window_name in ['square','rec','boxcar']:
        filter_ = np.asarray([int(abs(i) < .5) for i in sample_positions], dtype=np.float64)
        
    elif window_name in ['tri','triangular','bartlett']:
        filter_ = 1-2*abs(sample_positions)
        
    elif window_name in ['blackman']:
        filter_ = .42 + .5*np.cos(2*np.pi*sample_positions) + .08*np.cos(4*np.pi*sample_positions)
        
    elif window_name in ['blackharr']:
        filter_ = .35875 + .48829*np.cos(2*np.pi*sample_positions) + .14128*np.cos(4*np.pi*sample_positions) + \
            .01168*np.cos(6*np.pi*sample_positions)
        
    elif window_name in ['modblackharr']:
        filter_ = .35872 + .48832*np.cos(2*np.pi*sample_positions) + .14128*np.cos(4*np.pi*sample_positions) + \
            .01168*np.cos(6*np.pi*sample_positions)
        
    elif window_name in ['nuttall','nuttall12']:
        filter_ = .355768 + .487396*np.cos(2*np.pi*sample_positions) + .144232*np.cos(4*np.pi*sample_positions) + \
            .012604*np.cos(6*np.pi*sample_positions)
        
    elif window_name in ['nuttall20']:
        filter_ = 3/8 + 4/8*np.cos(2*np.pi*sample_positions) + 1/8*np.cos(4*np.pi*sample_positions)
        
    elif window_name in ['nuttall11']:
        filter_ = .40897 + .5*np.cos(2*np.pi*sample_positions) + .09103*np.cos(4*np.pi*sample_positions)
        
    elif window_name in ['nuttall02']:
        filter_ = .4243801 + .4973406*np.cos(2*np.pi*sample_positions) + .0782793*np.cos(4*np.pi*sample_positions)
        
    elif window_name in ['nuttall30']:
        filter_ = 10/32 + 15/32*np.cos(2*np.pi*sample_positions) + 6/32*np.cos(4*np.pi*sample_positions) + \
            1/32*np.cos(6*np.pi*sample_positions)
        
    elif window_name in ['nuttall21']:
        filter_ = .338946 + .481973*np.cos(2*np.pi*sample_positions) + .161054*np.cos(4*np.pi*sample_positions) + \
            .018027*np.cos(6*np.pi*sample_positions)
        
    elif window_name in ['nuttall03']:
        filter_ = .3635819 + .4891775*np.cos(2*np.pi*sample_positions) + .1365995*np.cos(4*np.pi*sample_positions) + \
            .0106411*np.cos(6*np.pi*sample_positions)
        
    elif window_name in ['gauss','truncgauss']:
        filter_ = np.exp(-18*sample_positions**2)
        
    elif window_name in ['wp2inp']:
        filter_ = np.exp(np.exp(-2*sample_positions)*25.*(1+2*sample_positions))
        filter_ = filter_/np.max(filter_)
    else:
        print("Error: unknown window function name ", window_name, " provided.")
        return None

    # List comprehension makes a mask to force values outside -.5 and .5 to zero
    mask = [int(abs(i) < .5) for i in sample_positions]
    filter_ *= mask
    return filter_