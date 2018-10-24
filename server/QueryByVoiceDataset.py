import librosa
import logging
import logging.config
import os
from abc import ABC, abstractmethod
from audioread import NoBackendError


class QueryByVoiceDataset(ABC):
    '''
    Abstract base class for an audio dataset for query-by-voice systems
    '''
    def __init__(self,
                 dataset_directory,
                 representation_directory,
                 model,
                 similarity_model_batch_size=None,
                 representation_batch_size=None):
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
            representation_batch_size: An integer or None. The maximum number
                of audio files to load during one batch of representation
                construction.
            similarity_model_batch_size: An integer or None. The maximum number
                of representations to load during one batch of model inference.
        '''
        # Setup logging
        parent_directory = os.path.dirname(os.path.abspath(__file__))
        logging.config.fileConfig(
            os.path.join(parent_directory, 'logging.conf'))
        self.logger = logging.getLogger('Dataset')

        self.dataset_directory = dataset_directory
        self.representation_directory = representation_directory
        self.similarity_model_batch_size = similarity_model_batch_size
        self.representation_batch_size = representation_batch_size
        self.model = model

        # Get all files in dataset
        audio_filenames = self._get_audio_filenames()
        try:
            # Create representation directory
            os.makedirs(self.representation_directory)
            logger.info('Representation directory not found. Building all \
                         representations from scratch')
            unrepresented = audio_filenames
        except OSError:
            # Find the files that don't have representation
            representation_handles = self._get_representation_handles()
            unrepresented = self._find_unrepresented(
                audio_filenames, representation_handles)

        # Build the representations and write them to the representation
        # directory
        self._build_representations(unrepresented)

    @abstractmethod
    def data_generator(self, query):
        '''
        Provides a generator that returns the necessary data for inference of
        a query-by-voice model. The generator yields the following:

            batch_query: A numpy array of length representation_batch_size. The
                chunks of the query to be compared with batch_representations.
                This may be windowed chunks of the query in the case that
                self.generate_pairs is True.
            batch_representations: A numpy array of length
                representation_batch_size. The chunks of representations to be
                compared to batch_query. This may be windowed chunks of the
                original audio in the case that self.generate_pairs is True.

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
    def _find_unrepresented(self, audio_filenames, representation_handles):
        '''
        Finds the list of audio files that do not have saved representations.

        Arguments:
            audio_filenames: A python list. All audio files in this dataset.
            representation_handles: A python list. The handles of all currently
                cached representations

        Returns:
            A python list.
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

    def _build_representations(self, audio_filenames):
        '''
        Constructs the audio representations and saves them to disk.

        Arguments:
            audio_filenames: A list. The filenames of the audio within
                dataset_directory that require representation.
        '''
        # Build a generator for reading in audio
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
                logger.warning('The file {} could not be decoded by any \
                    backend. Either no backends are available or each \
                    available backend failed to decode the \
                    file'.format(filename))
                continue

            # If we've successfully read a batch, yield the batch
            batch_size = self.similarity_model_batch_size
            if batch_size and len(audio) == batch_size:
                yield audio_list, sampling_rates, filenames
                audio_list = []
                sampling_rates = []
                filenames = []

        yield audio_list, sampling_rates, filenames
