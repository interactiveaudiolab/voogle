import audaugio
import logging
from abc import ABC, abstractmethod


class QueryByVoiceModel(ABC):
    '''
    Abstract base class for a query-by-voice machine learning model
    '''
    def __init__(self, uses_windowing, window_length, hop_length):
        '''
        QueryByVoiceModel constructor.

        Arguments:
            uses_windowing: A boolean. Indicates whether the model slices the
                representation
            window_length: A float. The window length in seconds. Unused if
                uses_windowing is False.
            hop_length: A float. The hop length between windows in seconds.
                Unused if uses_windowing is False.
        '''
        self.logger = logging.getLogger('Model')

        self.model = None
        self.uses_windowing = uses_windowing
        self.window_length = window_length
        self.hop_length = hop_length

        if self.uses_windowing:
            self.windower = audaugio.WindowingAugmentation(
                window_length=self.window_length,
                hop_size=self.hop_length,
                drop_last=False)

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
    def load_model(self, model_filepath):
        '''
        Loads the model weights from disk. Prepares the model to be able to
        make predictions.

        Arguments:
            model_filepath: A string. The path to the model weight file on
                disk.

        Returns:
            None
        '''
        pass

    @abstractmethod
    def predict(self, query, dataset):
        '''
        Runs model inference on the query.

        Arguments:
            query: An audio representation as defined by
                construct_representation. The user's vocal query.
            dataset: A python list of audio representations as defined by
                construct_representation. The dataset of potential matches for
                the user's query.

        Returns:
            A python list of floats. The similarity score of the query and each
                element in the dataset. The list order should be the same as
                in dataset.
        '''
        pass

    def _window(self, audio, sampling_rate):
        '''
        Chops the audio into windows of self.window_length seconds.

        Arguments:
            audio: A 1D numpy array. The audio to window.
            sampling_rate: An int. The sampling rate of the audio.

        Returns:
            A python list of equal-sized 1D numpy arrays.
        '''
        return self.windower.augment(audio, sampling_rate)
