from abc import ABC, abstractmethod


class QueryByVoiceModel(ABC):
    '''
    Abstract base class for a query-by-voice machine learning model
    '''
    def __init__(self):
        '''
        QueryByVoiceModel constructor.
        '''
        self.model = None

    @abstractmethod
    def get_name(self):
        '''
        Get the model name.

        Returns:
            A string.
        '''
        pass

    @abstractmethod
    def construct_representation(self, audio_list, sampling_rate, is_query):
        '''
        Constructs the audio representation used during inference. Audio
        files from the dataset are constructed only once and cached for
        later reuse.

        Arguments:
            audio_list: A python list of 1D numpy arrays. Each array represents
                one variable-length mono audio file.
            sampling_rate: The uniform sampling rate of all audio files in
                audio_list
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
