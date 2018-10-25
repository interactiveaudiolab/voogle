import logging
import numpy as np
import os
# from text.FilenameContains import FilenameContains


class Voogle(object):
    '''
    A query-by-voice system.
    '''

    def __init__(self, model, dataset, text_handler=None, matches=10):
        '''
        Voogle constructor

        Arguments:
            model: A QueryByVoiceModel. The model used to perform similarity
                calculations
            dataset: A python generator. The generator used to load audio
                representations for similarity ranking.
            text_handler: A TextHandler object. The model for determining if the
                user's text matches the audio text description.
            matches: An int. The number of matches to return during search.
        '''
        self.logger = logging.getLogger('Voogle')
        self.logger.debug('Initializing')

        self.model = model
        self.dataset = dataset
        self.text_handler = text_handler
        self.matches = matches

        self.logger.debug('Initialization complete')

    def search(self, query, sampling_rate, text_input=''):
        '''
        Search the dataset for the closest match to the given vocal query.

        Arguments:
            query: A 1D numpy array. The vocal query.
            sampling_rate: An integer. The sampling rate of the query.
            text_input: A string. Optional text input describing the target
                sound.

        Returns:
            A list. The names of audio files within the database sorted in
                descending order of similarity with the user query.
        '''
        # Construct query representation
        query = self.model.construct_representation(
            [query], [sampling_rate], is_query=True)

        # Retrieve the similarity measure between query and each dataset entry
        model_output = {}
        previous_filename = ''
        previous_index = 0
        generator = self.dataset.data_generator(query)
        for batch_query, batch_items, file_tracker in generator:

            # Run inference on this batch
            ranks = self.model.predict(batch_query, batch_items)

            # Determine the best score for each audio file
            for index, filename in file_tracker.items():
                if index != 0 and previous_filename != '':
                    max_file_rank = np.max(ranks[previous_index:index])
                    model_output = self._update_model_output(
                        model_output, previous_filename, max_file_rank)

                previous_filename = filename
                previous_index = index

            max_file_rank = np.max(ranks[previous_index:])
            model_output = self._update_model_output(
                model_output, previous_filename, max_file_rank)

        # Retrieve the top audio filenames
        match_list = sorted(model_output, key=model_output.get)[:self.matches]

        # Find the audio files also containing the user's text query
        text = text_input.lower()
        text_matches = [
            text and text in filename.lower() for filename in match_list]

        return match_list, text_matches

    def _update_model_output(self, model_output, filename, max_file_rank):
        if filename in model_output:
            model_output[filename] = max(
                model_output[filename], max_file_rank)
        else:
            model_output[filename] = max_file_rank
        return model_output
