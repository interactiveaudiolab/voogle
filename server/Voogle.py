import numpy as np
import os
from model.text.ContainsText import ContainsText
from log import get_logger


class Voogle(object):
    '''
    A query-by-voice system.
    '''

    def __init__(
        self,
        model,
        dataset,
        require_text_match,
        text_handler=ContainsText(),
        matches=10):
        '''
        Voogle constructor

        Arguments:
            model: A QueryByVoiceModel. The model used to perform similarity
                calculations
            dataset: A python generator. The generator used to load audio
                representations for similarity ranking.
            require_text_match: A boolean. If true ranking is performed only on
                dataset items that match the user's text query.
            text_handler: A TextHandler object. The model for determining if
                the user's text matches the audio text description.
            matches: An int. The number of matches to return during search.
        '''
        self.logger = get_logger('Voogle')

        self.logger.debug('Initializing')

        self.model = model
        self.dataset = dataset
        self.require_text_match = require_text_match
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
            Three equal-sized lists.
                - The names of audio files within the database sorted in
                    descending order of similarity with the user query.
                - A list of booleans indicating if the audio file text matches
                    the user's text query.
                - A list of float-valued similarities corresponding to the
                    similarity score of the audio file located at the same
                    index.
        '''
        # Construct query representation
        query = self.model.construct_representation(
            [query], [sampling_rate], is_query=True)

        # Seed the text handler with the user's text query
        self.text_handler.set_query_text(text_input)

        # Retrieve the similarity measure between query and each dataset entry
        model_output = {}
        previous_handle = ''
        previous_index = 0
        generator = self.dataset.data_generator(
            query, self.text_handler, self.require_text_match)
        for batch_query, batch_items, file_tracker in generator:

            # Run inference on this batch
            ranks = self.model.measure_similarity(batch_query, batch_items)

            # Determine the best score for each audio file
            for index, handle in file_tracker.items():
                if index != 0 and previous_handle != '':
                    max_file_rank = np.max(ranks[previous_index:index])
                    model_output = self._update_model_output(
                        model_output, previous_handle, max_file_rank)

                previous_handle = handle
                previous_index = index

            max_file_rank = np.max(ranks[previous_index:])
            model_output = self._update_model_output(
                model_output, previous_handle, max_file_rank)

        # Retrieve the top audio filenames
        match_list = sorted(model_output, key=model_output.get)[-self.matches:]
        match_list.reverse()
        filenames = [self.dataset.handle_to_filename(m) for m in match_list]

        # Find the audio files also containing the user's text query
        if self.require_text_match or not text_input:
            text_matches = [False] * len(match_list)
        else:
            text_features = [
                self.dataset.handle_to_text_features(m) for m in match_list]
            text_matches = [
                self.text_handler.is_match([t]) for t in text_features]

        # Retrieve the normalized similarity scores of the matches
        max_score = model_output[match_list[0]]
        similarity_scores = [model_output[m] / max_score for m in match_list]

        return filenames, text_matches, similarity_scores

    def _update_model_output(self, model_output, handle, max_file_rank):
        if handle in model_output:
            model_output[handle] = max(
                model_output[handle], max_file_rank)
        else:
            model_output[handle] = max_file_rank
        return model_output
