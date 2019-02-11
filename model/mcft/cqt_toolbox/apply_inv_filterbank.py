from __future__ import print_function,division

import numpy as np


def apply_inv_filterbank(cqt,inv_filter_bank,shift,sig_len,phasemode):
    # type: (list, list, numpy.ndarray, long, str) -> numpy.ndarray
    '''
    Input parameters: 
          cqt               : List of nonstationary Gabor coefficients
          inv_filterbank    : List of synthesis filters
          shift             : Ndarray of time shifts
          sig_len           : Length of the analyzed signal
          phasemode         : Can be set to (default is 'global')
                              - 'local'  : Zero-centered filtered used
                              - 'global' : Mapping function used (see reference)
    Output parameters:
          rec_signal        : Synthesized real-valued signal (Channels are stored 
                              in the columns) 

    This function synthesizes a real-valued signal by applying the 
    provided nonstationary Gabor filterbank to the cqt coefficients 
    positioned by the ndarray of shifts.

    Note that, due to the structure of the coefficient array in the real
    valued setting, all entries g{n} with N > length(c) will be ignored
    and assumed to be fully supported on the negative frequencies. 

    Let P(n)=sum_{l=1}^{n} shift(l), then the synthesis formula reads: 

                     N-1 
        fr_temp(l) = sum sum c{n}(m)g{n}[l-P(n)]*exp(-2*pi*i*(l-P(n))*m/M(n)),
                     n=0  m
    
    for l=0,...,Ls-1.  In practice, the synthesis formula is realized 
    by fft and overlap-add. To synthesize the negative frequencies, 
    fr_temp is truncated to length floor( Ls/2 )+1. Afterwards 
    ifftreal implicitly computes the hermite symmetric extension and 
    computes the inverse Fourier transform, i.e. fr = ifftreal(fr_temp).
  
    If a nonstationary Gabor frame was used to produce the coefficients 
    and inv_filterbank is a corresponding dual frame, this function should 
    perfectly reconstruct the originally analyzed signal to numerical precision.
    
    Multichannel output will save each channel in a column of rec_signal, such 
    that the output is of the shape (timeseries, num_channels).

    References:
      C. Schorkhuber, A. Klapuri, N. Holighaus, and M. Dorfler. A Matlab 
      Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy 
      Transforms.
 
      P. Balazs, M. Dörfler, F. Jaillet, N. Holighaus, and G. A. Velasco.
      Theory, implementation and applications of nonstationary Gabor Frames.
      J. Comput. Appl. Math., 236(6):1481-1496, 2011.
      
      G. A. Velasco, N. Holighaus, M. DAJrfler, and T. Grill. Constructing an
      invertible constant-Q transform with non-stationary Gabor frames.
      Proceedings of DAFX11, Paris, 2011. 

    See also:  gen_inv_filterbank, icqt

    Translation from MATLAB by: Trent Cwiok (cwiok@u.northwestern.edu)
                                Fatemeh Pishdadian (fpishdadian@u.northwestern.edu)
    '''
    # Input checking
    if len(cqt[0].shape) > 1:
        num_channels = cqt[0].shape[1]
    else:
        num_channels = 1
    num_filters = len(cqt)  

    # Variable initialization for reconstructed signal
    ctr_freqs = np.cumsum(shift)
    rec_sig_len = ctr_freqs[-1]
    ctr_freqs -= shift[0]
    rec_signal = np.zeros((int(rec_sig_len),int(num_channels)),dtype=np.complex128)

    # Apply the inv_filterbank to generate the reconstructed signal
    for i in range(num_filters):
        filter_len = len(inv_filter_bank[i])
        filter_range = ((ctr_freqs[i] + np.arange(-1*np.floor(filter_len/2),np.ceil(filter_len/2))) % rec_sig_len).astype(np.int32)

        temp = np.fft.fft(cqt[i],axis=0)*len(cqt[i])
        
        if phasemode == 'global':
            fsNewBins = len(cqt[i])
            fkBins = ctr_freqs[i]
            displace = fkBins - np.floor(fkBins/fsNewBins) * fsNewBins
            temp = np.roll(temp, -1*int(displace))

        # Apply the filter around the center frequency
        first_half = np.arange(len(temp)-np.floor(filter_len/2),len(temp))
        second_half = np.arange(np.ceil(filter_len/2))
        idx1 = np.concatenate((first_half,second_half))
        temp = temp[np.mod(idx1,len(temp)).astype(np.int32)]
        idx2 = np.concatenate((np.arange(filter_len-np.floor(filter_len/2),filter_len),np.arange(np.ceil(filter_len/2)))).astype(np.int32)
        rec_signal[filter_range,:] += (temp * inv_filter_bank[i][idx2]).reshape(len(temp),1)

    # Finish the synthesis of the signal with some arithmetic operations and an IFFT
    nyqBin = int(np.floor(sig_len/2))
    rec_signal[nyqBin+1:] = np.conj(rec_signal[nyqBin - (1- (sig_len%2)):0:-1])
    rec_signal = np.real(np.fft.ifft(rec_signal,axis=0))

    return rec_signal