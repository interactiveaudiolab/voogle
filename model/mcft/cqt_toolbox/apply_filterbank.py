from __future__ import print_function, division

import numpy as np 

def apply_filterbank(signal,filter_bank,shift,phasemode,bw_bins=None):
    # type: (numpy.ndarray, list, numpy.ndarray, str, numpy.ndarray) -> (list, long)
    '''
    Input parameters: 
        signal          : A real-valued signal -- multichannel
                          signals should be ndarray of shape
                          (len_of_signal, num_channels)
        filter_bank     : List of filters for each center
                          frequency of analysis
        shift           : Ndarray of frequency shifts
        phasemode       : 'local': zero-centered filtered used
                          'global': mapping function used (see cqt)
        **bw_bins       : Number of time channels 
                           If this is constant, the output is converted
                           to a matrix

    Output parameters:
        cqt             : Transform coefficients
        sig_len         : Original signal length

    **optional args 

    Given a (multichannel) signal, bank of filters, and frequency shifts,
    apply_filterbank applies the corresponding nonstationary Gabor filter
    to the real signal, using only the filters with at least partially
    supported on the positive frequencies. Let P(n)=sum_{l=1}^{n} shift(l), 
    then the output cqt = apply_filterbank(signal,filter_bank,shift,bw_bins)
    is a cell array with 

              Ls-1                                      
      c{n}(m)= sum fft(f)(l)*conj(g{n}(l-P(n)))*exp(2*pi*i*(l-P(n))*m/M(n))
               l=0                                        

    where m runs from 0 to M(n)-1 and n from 1 to N, where
    g{N} is the final filter at least partially supported on the positive
    frequencies. All filters in the bank and shift that are completely
    supported on the negative frequencies are ignored.

    See also:  nsigtf_real, nsdual, nstight

    Translation from MATLAB by: Trent Cwiok (cwiok@u.northwestern.edu)
                                Fatemeh Pishdadian (fpishdadian@u.northwestern.edu)

    '''
    # Unpack the signal length and num of channels
    sig_len,num_channels = signal.shape

    # Setup some useful variables for computation later on
    num_filters = len(shift)
    if bw_bins is None:
        bw_bins = np.zeros(num_filters)
        for i in range(num_filters):
            bw_bins[i] = len(filter_bank[i])

    if bw_bins.size == 1:
        bw_bins = bw_bins[0]*np.ones(num_filters)

    signal = np.fft.fft(signal,axis=0)

    # Convert from distance between center freqs to positions
    ctr_freqs = np.cumsum(shift)-shift[0]

    # Padding for scale frames
    zpad_len = np.sum(shift)-sig_len
    padding = np.zeros((int(zpad_len),int(num_channels)))
    signal = np.vstack((signal,padding))

    filter_lens = np.zeros(len(filter_bank))
    for i in range(len(filter_bank)):
        filter_lens[i] = len(filter_bank[i])

    # Number of filters determined by the last center freq position greater than the signal length
    num_filters = [ctr_freqs[i] - np.floor(filter_lens[i]/2) <= (sig_len+zpad_len)/2 for i in range(len(ctr_freqs))]
    num_filters = np.nonzero(num_filters)[0][-1]
    cqt = []
    
    # Applying the filters at the correct center freqs
    for i in range(num_filters+1):
        # Create the lists of indices at which points of the filters will be applied to the signal
        idx = np.concatenate((np.arange(np.ceil(filter_lens[i]/2),filter_lens[i]),np.arange(np.ceil(filter_lens[i]/2))))
        filter_range = ((ctr_freqs[i] + np.arange(-1*np.floor(filter_lens[i]/2),np.ceil(filter_lens[i]/2))) % sig_len+zpad_len)
        idx,filter_range = (idx.astype(np.int32), filter_range.astype(np.int32))
        
        # Create the cqt coefficients
        if bw_bins[i] < filter_lens[i]:
            # This case involves aliasing (non-painless case)
            col = np.ceil(filter_lens[i]/bw_bins[i])
            temp = np.zeros((col*bw_bins[i], num_channels))

            slice_one = np.arange((temp.shape[0]-np.floor(filter_lens[i]/2)),temp.shape[0],dtype=np.int32)
            slice_two = np.arange(np.ceil(filter_lens[i]/2),dtype=np.int32)
            temp[np.concatenate((slice_one,slice_two)),:] = signal[filter_range,:] * filter_bank[i][idx]
            
            temp = np.reshape(temp,(bw_bins[i],col,num_channels), dtype=np.complex128)
            cqt.append(np.squeeze(np.fft.ifft(np.sum(temp, axis=1))))

        else:
            # No aliasing here
            temp = np.zeros((int(bw_bins[i]),num_channels), dtype=np.complex128)
            slice_one = np.arange((temp.shape[0]-np.floor(filter_lens[i]/2)),temp.shape[0],dtype=np.int32)
            slice_two = np.arange(np.ceil(filter_lens[i]/2),dtype=np.int32)
            temp[np.concatenate((slice_one,slice_two)),:] = signal[filter_range,:] * np.reshape(filter_bank[i][idx],(len(filter_bank[i]),1))

            if phasemode == 'global':
                fsNewBins = bw_bins[i]
                fkBins = ctr_freqs[i]
                displace = fkBins - np.floor(fkBins/fsNewBins) * fsNewBins
                temp = np.roll(temp, int(displace))
        
            cqt.append(np.fft.ifft(temp, axis=0))
            
    # If coefficients are all teh same length, reshape the list into an ndarray
    if np.max(bw_bins) == np.min(bw_bins):
        cqt = np.asarray(cqt)
        cqt = np.reshape(cqt, (int(bw_bins[0]),int(num_filters+1),int(num_channels)))

    return cqt, sig_len