import librosa
import numpy as np
import os
from abc import ABC, abstractmethod
from audioread import NoBackendError
from log import get_logger
from progress.bar import Bar


class QueryByVoiceDataset(ABC):
    '''
    Abstract base class for an audio dataset for query-by-voice systems
    '''
    def __init__(self,
                 dataset_directory,
                 representation_directory,
                 model,
                 measure_similarity_batch_size,
                 construct_representation_batch_size):
        '''
        Dataset constructor.

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
        self.logger = get_logger('Dataset')

        self.dataset_directory = dataset_directory
        self.representation_directory = representation_directory
        self.model = model
        self.measure_similarity_batch_size = measure_similarity_batch_size
        self.construct_representation_batch_size = \
            construct_representation_batch_size

        if self._dataset_directory_empty():
            self.logger.error('No dataset found!')
            raise FileNotFoundError('No dataset found!')

        if (self._representation_directory_empty() or
            self._dataset_directory_was_updated() or
            (self.model.parametric_representation and
             self._model_was_updated())):
            # Build the representations and write them to the representation
            # directory
            self.logger.info('Building all representations from scratch')
            self._build_representations()

    @abstractmethod
    def data_generator(self, query):
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
            file_tracker: A dict. Maps the representation handles to their
                starting index within a batch.

        Arguments:
            query: A numpy array. The audio representation of the user's query.

        Returns:
            A python generator.
        '''
        pass

    @abstractmethod
    def generator_feedback(self, model_output):
        '''
        Provides the generator with feedback after one batch of inference. Can
        be used to implement tree-based search through the representations.

        Arguments:
            model_output: A python list. The float-valued similarity scores
                output by the model.
        '''
        pass

    @abstractmethod
    def handle_to_filename(self, handle):
        '''
        Given an audio representation handle, returns the original audio
        filename

        Arguments:
            handle: The audio representation handle as defined by the dataset.

        Returns:
            A string. The path to the audio file relative to dataset_directory.
        '''
        pass

    @abstractmethod
    def handle_to_text_features(self, handle):
        '''
        Given a representation handle, returns the text features associated with
        the underlying audio file

        Arguments:
            handle: A representation handle

        Returns:
            A list of strings
        '''
        pass

    @abstractmethod
    def _get_audio_filenames(self):
        '''
        Retrieves a list of all audio fileanames in this dataset. Filenames must
        be relative to dataset_directory (i.e., the statement

            os.path.isfile(os.path.join(self.dataset_directory, filename))

        must return True).

        Returns:
            A python list.
        '''
        pass

    @abstractmethod
    def _get_representation_handles(self):
        '''
        Retrieves a list of dataset-specific handles for accessing the cached
        audio representations (e.g., filenames relative to
        representation_directory or filenames + indexes into a hdf5 file).

        Returns:
            A python list.
        '''
        pass

    @abstractmethod
    def _load_representations(self, handles):
        '''
        Loads and returns the audio representations.

        Arguments:
            handles: A python list. The representation handles that must be
                loaded.

        Returns:
            A python list.
        '''
        pass

    @abstractmethod
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
        pass

    def _build_representations(self):
        '''
        Constructs the audio representations and saves them to disk.

        Arguments:
            audio_filenames: A list. The filenames of the audio within
                dataset_directory that require representation.
        '''
        try:
            # Create representation directory
            os.makedirs(self.representation_directory)
            self.logger.info('Created representation directory')
        except OSError:
            pass

        # Build a generator for reading in audio
        audio_filenames = self._get_audio_filenames()
        generator = self._build_audio_generator(audio_filenames)

        # Build audio representations in batches
        for audio, sampling_rates, filenames in generator:
            representations = self.model.construct_representation(
                audio, sampling_rates, is_query=False)
            self._save_representations(representations, filenames)


    def _build_audio_generator(self, audio_filenames):
        audio_list = []
        sampling_rates = []
        filenames = []
        for filename in audio_filenames:
            try:
                # read the file in as an audio file
                filepath = os.path.join(self.dataset_directory, filename)
                audio, sampling_rate = librosa.load(filepath, sr=None)
                audio_list.append(audio)
                sampling_rates.append(sampling_rate)
                filenames.append(filename)
            except NoBackendError:
                # either non-audio file or bad audioread setup
                self.logger.warning('The file {} could not be decoded by any \
                    backend. Either no backends are available or each \
                    available backend failed to decode the \
                    file'.format(filename))
                continue

            # If we've successfully read a batch, yield the batch
            batch_size = self.measure_similarity_batch_size
            if batch_size and len(audio) == batch_size:
                yield audio_list, sampling_rates, filenames
                audio_list = []
                sampling_rates = []
                filenames = []

        yield audio_list, sampling_rates, filenames

    def _dataset_directory_empty(self):
        # Build the dataset directory if it does not exist
        try:
            os.makedirs(os.path.dirname(self.dataset_directory))
            self.logger.info('Created audio directory')
            return True
        except OSError:
            return False

    def _dataset_directory_was_updated(self):
        # Timestamp of representation directory
        timestamp = os.path.getmtime(self.representation_directory)

        # Has any file been updated?
        files = self._get_audio_filenames()
        files = [os.path.join(self.dataset_directory, file) for file in files]
        file_was_updated = any([
            timestamp < os.path.getmtime(file) for file in files])

        # Has the directory itself been updated?
        directory_was_updated = (
            timestamp < os.path.getmtime(self.dataset_directory))

        result = file_was_updated or directory_was_updated

        if result:
            self.logger.info('Found updated dataset directory.')
        return result

    def _linear_data_generator(self, query, text_handler, require_text_match):
        '''
        Provides a generator that iterates linearly through all points in the
        dataset during inference. The generator yields the following:

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
            file_tracker: A dict. Maps the representation handles to their
                starting index within a batch.

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
        handles = self._get_representation_handles()

        # Reduce the set of representation handles to only those with file
        # text data matching the user's text query
        if require_text_match:
            handles = [h for h in handles if text_handler.is_match(
                [self.handle_to_text_features(h)])]

        # If no batch size is set, load entire dataset of representations
        if (self.construct_representation_batch_size):
            max_batch_size = self.construct_representation_batch_size
        else:
            max_batch_size = len(handles)

        start = 0
        end = len(handles)
        while start < end:
            batch_handles = handles[start:min(start+max_batch_size, end)]
            representations = self._load_representations(batch_handles)

            # Handle pairwise comparisons
            if self.model.uses_windowing:
                for batch in self._pairwise_batch_generator(
                    query, representations, batch_handles):
                    yield batch
            else:
                batch_query = np.repeat(
                    np.array(query), len(representations), axis=0)
                batch_size = min(len(representations), max_batch_size)
                file_tracker = {i : batch_handles[i] for i in range(batch_size)}
                yield batch_query, np.array(representations), file_tracker

            start += max_batch_size

    def _linear_generator_feedback(self, model_output):
        '''
        The linear generator defined in _linear_data_generator does not
        incorporate model feedback, so this is a no-op.

        Arguments:
            model_output: A python list. The float-valued similarity scores
                output by the model.
        '''
        pass

    def _model_was_updated(self):
        result = (os.path.getmtime(self.representation_directory) <
                  os.path.getmtime(self.model.model_filepath))
        if result:
            logger.info('Found updated model weights.')
        return result

    def _pairwise_batch_generator(self, query, representations, handles):
        '''
        Provides a generator that returns batches of pairs of windows of the
        query and representation. Also returns a file tracker that tracks the
        indices in the returned batch that belong to each audio file.

        Arguments:
            query: A numpy array. The audio representation of the user's query.
            representations: A python list. The windowed representations.
            handles: A python list. The handles corresponding to each
                representation.

        Returns:
            A python generator.
        '''
        batch_size = self.construct_representation_batch_size
        num_query_windows = len(query)

        batch_representations = []
        batch_query = []
        file_tracker = {}
        index = 0
        for representation, handle in zip(representations, handles):
            num_representation_windows = len(representation)
            num_pairs = num_query_windows * num_representation_windows

            # Cartesian product of query and representation windows
            # TODO: use views to reduce memory storage
            for i in range(num_representation_windows):
                for j in range (num_query_windows):
                    batch_representations.append(representation[i])
                    batch_query.append(query[j])

            # All above pairs belong to one representation. Mark the start point
            # of that representation in the batch.
            file_tracker[index] = handle
            index += num_pairs

            if batch_size and index >= batch_size:
                yield (
                   np.array(batch_query[:batch_size]),
                   np.array(batch_representations[:batch_size]),
                   file_tracker)

                # If we have more than batch_size pairs, yield the others on the
                # next batch
                batch_query = batch_query[batch_size:]
                batch_representations = batch_representations[batch_size:]
                file_tracker = {} if index == batch_size else {0 : filename}
                index -= batch_size

        # Yield the last batch, or all pairs if batch_size == None
        if index != 0:
            yield (
                np.array(batch_query),
                np.array(batch_representations),
                file_tracker)

    def _representation_directory_empty(self):
        try:
            # Create representation directory
            os.makedirs(self.representation_directory)
            self.logger.info('Created representation directory')
            return True
        except OSError:
            result = len(os.listdir(self.representation_directory)) == 0
            if result:
                self.logger.info('Found empty representation directory.')
            return result
