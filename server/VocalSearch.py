import logging
import logging.config
import numpy as np
import os

logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf'))
logger = logging.getLogger('VocalSearch')


class VocalSearch(object):
    '''
    A query-by-voice system.
    '''

    def __init__(self, model, dataset):
        '''
        VocalSearch constructor

        Arguments:
            model: A QueryByVoiceModel. The model used to perform similarity
                calculations
            dataset: A python generator. The generator used to load audio
                representations for similarity ranking.
        '''
        logger.debug('Initializing')

        self.model = model
        self.dataset = dataset

        logger.debug('Initialization complete')

    def search(self, query, sampling_rate):
        '''
        Search the dataset for the closest match to the given vocal query.

        Arguments:
            query: A 1D numpy array. The vocal query.
            sampling_rate: An integer. The sampling rate of the query.

        Returns:
            A list. The names of audio files within the database sorted in
                descending order of similarity with the user query.
        '''
        # Construct query representation
        query = self.model.construct_representation(
            [query], [sampling_rate], True)

        # Retrieve the similarity measure between query and each dataset entry
        model_output = []
        filenames = []
        for batch, batch_filenames in self.dataset.data_generator():
            model_output.append(self.model.predict(query, batch))
            filenames += batch_filenames
        model_output = np.array(model_output).flatten()

        # Determine ranking
        sorted_index = np.argsort(model_output.flatten())[::-1]

        # Sort the files by similarity rank
        return list(np.array(filenames)[sorted_index])
