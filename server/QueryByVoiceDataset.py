from abc import ABC, abstractmethod


class QueryByVoiceDataset(ABC):
    '''
    Abstract base class for an audio dataset for query-by-voice systems
    '''
    def __init__(self,
                 dataset_directory,
                 representation_directory,
                 similarity_model_batch_size,
                 representation_batch_size,
                 model):
        '''
        Dataset constructor.

        Arguments:
            dataset_directory: A string. The directory containing the dataset.
            representation_directory: A string. The directory containing the
                pre-constructed representations. If the directory does not
                exist, it will be created along with all audio representations
                for this dataset.
            representation_batch_size: An integer or None. The maximum number
                of audio files to load during one batch of representation
                construction.
            similarity_model_batch_size: An integer or None. The maximum number
                of representations to load during one batch of model inference.
            model: A QueryByVoiceModel. The model to be used in representation
                construction.
        '''
        self.dataset_directory = dataset_directory
        self.representation_directory = representation_directory
        self.similarity_model_batch_size = similarity_model_batch_size
        self.representation_batch_size = representation_batch_size
        self.model = model

    @abstractmethod
    def data_generator(self):
        '''
        Provides a generator for loading audio representations.

        Returns:
            A python generator.
        '''
        pass
