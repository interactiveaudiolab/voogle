import librosa
import math
import numpy as np
import os
from model.mcft.cqt_toolbox.cqt import cqt
from model.mcft.mcft_toolbox.mcft import cqt_to_mcft
from model.mcft.mcft_toolbox.spectro_temporal_fbank import (
    filt_default_centers, gen_fbank_scale_rate)
from model.QueryByVoiceModel import QueryByVoiceModel
import pickle
from scipy import spatial
import time


class MCFT(QueryByVoiceModel):
    '''
    A MCFT feature extractor for query-by-voice applications.

    citation: Fatemeh Pishdadian and Bryan Pardo. “Multi-resolution Common
        Fate Transform,” IEEE/ACM Transactions on Audio, Speech, and Language
        Processing, 2018
    '''

    def __init__(
        self,
        model_filepath,
        parametric_representation=False,
        uses_windowing=True,
        window_length=2.0,
        hop_length=1.0):
        '''
        MCFT model constructor.

        Arguments:
            model_filepath: A string. The path to the model weight file on
                disk.
            parametric_representation: A boolen. True if the audio
                representations depend on the model weights.
            uses_windowing: A boolean. Indicates whether the model slices the
                representation
            window_length: A float. The window length in seconds. Unused if
                uses_windowing is False.
            hop_length: A float. The hop length between windows in seconds.
                Unused if uses_windowing is False.
        '''
        super().__init__(
            model_filepath,
            parametric_representation,
            uses_windowing,
            window_length,
            hop_length)
        self.filter_bank = np.array([])

    def construct_representation(self, audio_list, sampling_rates, is_query):
        '''
        Constructs the audio representation used during inference. Audio
        files from the dataset are constructed only once and cached for
        later reuse.

        Arguments:
            audio_list: A python list of 1D numpy arrays. Each array represents
                one variable-length mono audio file.
            sampling_rate: A python list of ints. The corresponding sampling
                rate of each element of audio_list.
            is_query: A boolean. True only if audio is a user query.

        Returns:
            A python list of audio representations. The list order should be
                the same as in audio_list.
        '''
        representations = []
        for audio, sampling_rate in zip(audio_list, sampling_rates):

            new_sampling_rate = 8000
            audio = librosa.resample(audio, sampling_rate, new_sampling_rate)

            if self.uses_windowing:
                windows = self._window(audio, new_sampling_rate)
            else:
                windows = [
                    librosa.util.fix_length(
                        audio, self.window_length * new_sampling_rate)]

            representation = []
            for window in windows:
                if not self.filter_bank.any():
                    self.filter_bank = self._make_filter_bank(
                        window, new_sampling_rate)

                start = time.time()
                query_cqt_mag = self._compute_cqt(window, new_sampling_rate)
                mcft_out = cqt_to_mcft(query_cqt_mag, self.filter_bank)
                features = np.mean(np.abs(mcft_out), axis=(2, 3))
                end = time.time()
                print('time: {}'.format(end - start))
                representation.append(features)

            # normalize to zero mean and unit variance
            representation = np.array(representation)
            representations.append(representation)

        return representations

    def measure_similarity(self, query, items):
        '''
        Runs model inference on the query.

        Arguments:
            query: A numpy array. An audio representation as defined by
                construct_representation. The user's vocal query.
            items: A numpy array. The audio representations as defined by
                construct_representation. The dataset of potential matches for
                the user's query.

        Returns:
            A python list of floats. The similarity score of the query and each
                element in the dataset. The list order should be the same as
                in dataset.
        '''
        # run model inference
        self.logger.debug('Running inference')
        simlarities = []
        for q, i in zip(query, items):
            sim = []
            for window in q:
                sim.append(
                    1 - spatial.distance.cosine(window.flatten(), i.flatten()))
            simlarities.append(np.max(np.array(sim)))

        return np.array(simlarities)

    def _compute_cqt(self, query, sampling_rate):
        # cqt parameters
        fmin = 27.5*2**(0/12)
        fmax = 27.5*2**(87/12)
        fres = 24
        gamma = 0
        print(query.shape, sampling_rate)
        cqt_results = cqt(query, fres, sampling_rate, fmin, fmax, gamma=gamma)
        return np.abs(cqt_results['cqt'])

    def _compute_filter_bank(self, query, sampling_rate):
        fres = 24
        query_cqt_mag = self._compute_cqt(query, sampling_rate)
        num_freq_bin, num_time_frame = np.shape(query_cqt_mag)

        # filterbank parameters
        query_dur = len(query)/sampling_rate
        scale_res, rate_res = 1, 8
        samprate_spec = fres
        samprate_temp = np.floor(num_time_frame/query_dur)

        scale_nfft, rate_nfft = num_freq_bin, num_time_frame

        scale_nfft = int(2**np.ceil(np.log2(scale_nfft)))
        rate_nfft = int(2**np.ceil(np.log2(rate_nfft)))

        scale_params = (scale_res, scale_nfft, samprate_spec)
        rate_params = (rate_res, rate_nfft, samprate_temp)
        print(scale_params, rate_params)
        scale_ctrs, rate_ctrs = filt_default_centers(scale_params, rate_params)

        print(scale_ctrs, rate_ctrs)

        time_const = 1
        filt_params = {
            'samprate_spec': samprate_spec,
            'samprate_temp': samprate_temp,
            'time_const': time_const
        }

        _, fbank_sr_domain = gen_fbank_scale_rate(
            scale_ctrs, rate_ctrs, scale_nfft, rate_nfft, filt_params)

        return fbank_sr_domain

    def _load_model(self):
        '''
        Loads the model weights from disk. Prepares the model to be able to
        make predictions.
        '''
        pass

    def _make_filter_bank(self, query, sampling_rate):
        try:
            with open(self.model_filepath, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return self._compute_filter_bank(query, sampling_rate)
