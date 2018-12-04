import librosa
import numpy as np
import os
from abc import ABC, abstractmethod
from log import get_logger


class QueryByVoiceModel(ABC):
    '''
    Abstract base class for a query-by-voice machine learning model
    '''
    def __init__(
        self,
        model_filepath,
        parametric_representation,
        uses_windowing,
        window_length,
        hop_length):
        '''
        QueryByVoiceModel constructor.

        Arguments:
            model_filepath: A string. The path to the model weight file on
                disk.
            parametric_representation: A boolen. True if the audio
                representations depend on the model weights.
            uses_windowing: A boolean. Indicates whether the model slices the
                representation into fixed-length time windows.
            window_length: A float. The window length in seconds. Unused if
                uses_windowing is False.
            hop_length: A float. The hop length between windows in seconds.
                Unused if uses_windowing is False.
        '''
        self.logger = get_logger('Model')

        self.model = None
        self.model_filepath = model_filepath
        self.parametric_representation = parametric_representation
        self.uses_windowing = uses_windowing
        self.window_length = window_length
        self.hop_length = hop_length

        self._load_model()

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def _load_model(self):
        '''
        Loads the model weights from disk. Prepares the model to be able to
        make measure_similarityions.
        '''
        pass

    def _window(self, audio, sampling_rate):
        '''
        Chops the audio into windows of self.window_length seconds.

        Arguments:
            audio: A 1D numpy array. The audio to window.
            sampling_rate: An int. The sampling rate of the audio.

        Returns:
            A 2D numpy array of shape (windows, window_samples)
        '''
        window_samples = int(self.window_length * sampling_rate)
        hop_samples = int(self.hop_length * sampling_rate)

        if audio.shape[0] < window_samples:
            window = librosa.util.fix_length(audio, window_samples)
            return np.expand_dims(window, axis=0)
        else:
            return librosa.util.frame(audio, window_samples, hop_samples).T
