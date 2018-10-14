from abc import ABC, abstractmethod


class QueryByVoiceDataset(ABC):
    '''
    Abstract base class for an audio dataset for query-by-voice systems
    '''

    @abstractmethod
    def data_generator(self, similarity_model_batch_size=None,
                       reprsentation_batch_size=None):
        '''
        Provides a generator for loading audio representations.

        Arguments:
            similarity_model_batch_size: An integer or None. The maximum number
                of audio representations to return upon each call to the
                generator. If None, all representations are returned.
            reprsentation_batch_size: An integer or None. The maximum number of
                audio files to load into memory at once during representation
                construction. If None, all audio files are loaded and processed
                at once.
            model: A QueryByVoiceModel. The model to be used in representation
                construction.

        Returns:
            A python generator.
        '''
        pass
