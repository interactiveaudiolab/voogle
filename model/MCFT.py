import librosa
import numpy as np
import os
from model.mcft.mcft_toolbox.mcft import mcft
from model.QueryByVoiceModel import QueryByVoiceModel
from scipy import spatial

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
        window_length=4.0,
        hop_length=2.0):
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
        # cqt parameters
        fmin = 27.5*2**(0/12)
        fmax = 27.5*2**(87/12)
        fres = 24
        gamma = 10

        representations = []
        for audio, sampling_rate in zip(audio_list, sampling_rates):

            if self.uses_windowing:
                windows = self._window(audio, sampling_rate)
            else:
                windows = [
                    librosa.util.fix_length(
                        audio, self.window_length * sampling_rate)]

            representation = []
            for window in windows:
                # construct the mcft of the signal
                cqt_params_in = {
                    'samprate_sig': sampling_rate,
                    'fmin': fmin,
                    'fmax': fmax,
                    'fres': fres,
                    'gamma': gamma
                }
                mcft_out, _, _, _, _ = mcft(window, cqt_params_in)
                features = np.mean(mcft_out, axis=(0, 1))
                representation.append(features)

            # normalize to zero mean and unit variance
            representation = np.array(representation)
            # representation = self._normalize(representation).astype('float32')
            # representation = np.expand_dims(representation, axis=1)
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
        if not self.model:
            raise RuntimeError('No model loaded during call to \
                               measure_similarity.')

        # run model inference
        self.logger.debug('Running inference')
        simlarities = []
        for q, i in zip(query, items):
            simlarities.append(1 - spatial.distance.cosine(q, i))

        return np.array(simlarities)

    def _load_model(self):
        '''
        Loads the model weights from disk. Prepares the model to be able to
        make predictions.
        '''
        pass
