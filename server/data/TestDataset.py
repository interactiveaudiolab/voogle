import numpy as np
import os
from data.QueryByVoiceDataset import QueryByVoiceDataset

class TestDataset(QueryByVoiceDataset):
    '''
    A small dataset for testing query-by-voice systems
    '''

    def __init__(self,
                 dataset_directory,
                 representation_directory,
                 model,
                 measure_similarity_batch_size=None,
                 construct_representation_batch_size=None):
        '''
        TestDataset constructor.

        Arguments:
            dataset_directory: A string. The directory containing the dataset.
            representation_directory: A string. The directory containing the
                pre-constructed representations. If the directory does not
                exist, it will be created along with all audio representations
                for this dataset.
            model: A QueryByVoiceModel. The model to be used in representation
                construction.
            measure_similarity_batch_size: An integer or None. The maximum
                number of representations to load during one batch of model
                inference.
            construct_representation_batch_size: An integer or None. The maximum
                number of audio files to load during one batch of representation
                construction.
        '''
        super(TestDataset, self).__init__(
            dataset_directory,
            representation_directory,
            model,
            measure_similarity_batch_size,
            construct_representation_batch_size)

    def data_generator(self, query, text_handler, require_text_match):
        '''
        Provides a generator that returns the necessary data for inference of
        a query-by-voice model. The generator yields the following:

            batch_query: A numpy array of length
                construct_representation_batch_size. The chunks of the query to
                be compared with batch_representations. This may be windowed
                chunks of the query in the case that self.generate_pairs is
                True.
            batch_representations: A numpy array of length
                construct_representation_batch_size. The chunks of
                representations to be compared to batch_query. This may be
                windowed chunks of the original audio in the case that
                self.generate_pairs is True.
            file_tracker: A dict of (String, int). Maps the audio filenames to
                their starting index within a batch.

        Arguments:
            query: A numpy array. The audio representation of the user's query.
            text_handler: A TextHandler. Determines whether a file's text
                information is compatible with the user's text query.
            require_text_match: A boolean. If true, only representations of
                audio files with text data matching the user's text query are
                provided by the generator.

        Returns:
            A python generator.
        '''
        return self._linear_data_generator(
            query, text_handler, require_text_match)

    def generator_feedback(self, model_output):
        '''
        Provides the generator with feedback after one batch of inference. Can
        be used to implement tree-based search through the representations.

        Arguments:
            model_output: A python list. The float-valued similarity scores
                output by the model.
        '''
        self._linear_generator_feedback(model_output)

    def handle_to_filename(self, handle):
        '''
        Given an audio representation handle, returns the original audio
        filename

        Arguments:
            handle: The audio representation handle as defined by the dataset.

        Returns:
            A string. The path to the audio file relative to dataset_directory.
        '''
        print(handle, handle.rsplit('.', 1)[0])
        return handle.rsplit('.', 1)[0]

    def handle_to_text_features(self, handle):
        '''
        Given a representation handle, returns the text features associated with
        the underlying audio file

        Arguments:
            handle: A representation handle

        Returns:
            A list of strings
        '''
        # The test dataset handle is just the filename
        return self.handle_to_filename(handle)

    def _get_audio_filenames(self):
        '''
        Retrieves a list of all audio fileanames in this dataset. Filenames must
        be relative to dataset_directory (i.e., the statement

            os.path.isfile(os.path.join(self.dataset_directory, filename))

        must return True).

        Returns:
            A python list.
        '''
        return sorted(os.listdir(self.dataset_directory))

    def _get_representation_handles(self):
        '''
        Retrieves a list of dataset-specific handles for accessing audio
        representations (e.g., filenames relative to representation_directory or
        filenames + indexes into a hdf5 file).

        Returns:
            A python list.
        '''
        return sorted(os.listdir(self.representation_directory))

    def _load_representations(self, handles):
        '''
        Loads and returns the audio representations.

        Arguments:
            handles: A python list. The representation handles that must be
                resolved.

        Returns:
            A python list. Representations should be in the same order as the
                handles
        '''
        representations = []
        for handle in handles:
            filepath = os.path.join(self.representation_directory, handle)
            representation = np.load(filepath)
            representations.append(representation)
        return representations

    def _save_representations(self, representations, filenames):
        '''
        Saves the audio representations to disk.

        Arguments:
            representations: A python list. The list of audio representations
                to save.
            filenames:
                A python list. The corresponding list of audio filenames (i.e.,
                    representations[i] is the audio representation of
                    filenames[i]).
        '''
        # Save each representation as its own .npy file
        for representation, filename in zip(representations, filenames):
            filepath = os.path.join(
                self.representation_directory, filename)
            np.save(filepath + '.npy', representation)

