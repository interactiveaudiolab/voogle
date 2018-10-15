import librosa
import logging
import numpy as np
import os
from audioread import NoBackendError
from QueryByVoiceDataset import QueryByVoiceDataset


class TestDataset(QueryByVoiceDataset):
    '''
    A small dataset for testing query-by-voice systems
    '''

    def __init__(self, dataset_directory, representation_directory):
        '''
        TestDataset constructor.

        Arguments:
            dataset_directory: A string. The directory containing the dataset.
            representation_directory: A string. The directory containing the
                pre-constructed representations. If the directory does not
                exist, it will be created along with all audio representations
                for this dataset.
        '''
        super(TestDataset, self).__init__()
        self.dataset_directory = dataset_directory
        self.representation_directory = representation_directory

    def data_generator(self, model, similarity_model_batch_size=None,
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
        # Get all files in dataset
        audio_filenames = sorted(os.listdir(self.dataset_directory))
        try:
            # Find the files that don't have representation
            representation_filenames = sorted(
                os.listdir(self.representation_directory))
            unrepresented = self._find_audio_without_representation(
                audio_filenames, representation_filenames)
        except OSError:
            # Create representation directory
            logging.info('Representation directory not found. Building all' +
                         'representations from scratch')
            os.makedirs(self.representation_directory)
            representation_filenames = []
            unrepresented = audio_filenames

        # Build the representations and write them to the representation
        # directory
        self._build_representations(
            unrepresented, reprsentation_batch_size, model)

        # Construct the generator for the audio representations
        return self._build_representation_generator(
            similarity_model_batch_size)

    def _find_audio_without_representation(self, audio_filenames,
                                           representation_filenames):
        unrepresented = []
        non_corresponding = []
        i = 0
        j = 0
        while i < len(audio_filenames) and j < len(representation_filenames):
            audio = audio_filenames[i]
            representation = representation_filenames[j].rsplit('.', 1)[0]

            # Audio file has not been processed
            if audio < representation:
                unrepresented.append(audio)
                i += 1

            # Representation does not correspond with any audio file
            elif audio > representation:
                non_corresponding.append(representation)
                j += 1
            # Found a corresponding representation for the audio file
            else:
                i += 1
                j += 1

        # Audio files remain that have not been processed
        if i < len(audio_filenames):
            unrepresented += audio_filenames[i:]

        # Representations remain that do not have corresponding audio
        elif j < len(representation_filenames):
            for representation in representation_filenames[j:]:
                non_corresponding.append(representation.rsplit('.', 1)[0])

        # Report a list of bad representations
        if non_corresponding:
            logging.warning('Found representations not corresponding to any ' +
                            'known audio file: {}'.format(non_corresponding))

        return unrepresented

    def _build_representations(self, audio_filenames, batch_size, model):
        # Build a generator for reading in audio
        generator = self._build_audio_generator(audio_filenames, batch_size)

        # Build audio representations in batches
        for audio, sampling_rates, filenames in generator:
            representations = model.construct_representation(
                audio, sampling_rates, is_query=False)

            # Save each representation as its own .npy file
            # TODO: this isn't very efficient for reading. Consider hdf5.
            for representation, filename in zip(representations, filenames):
                filepath = os.path.join(
                    self.representation_directory, filename)
                np.save(filepath + '.npy', representation)

    def _build_representation_generator(self, batch_size):
        representations = []
        filenames = []
        for filename in os.listdir(self.representation_directory):

            # Read in a representation
            filepath = os.path.join(self.representation_directory, filename)
            representation = np.load(filepath)
            representations.append(representation)
            filenames.append(filename.rsplit('.', 1)[0])

            # If we've successfully read a batch, yield the batch
            if batch_size and len(representations) == batch_size:
                yield representations
                representations = []

        yield representations, filenames

    def _build_audio_generator(self, audio_filenames, batch_size):
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
                logging.warning('The file {} could not be decoded by any \
                    backend. Either no backends are available or each \
                    available backend failed to decode the \
                    file'.format(filename))
                continue

            # If we've successfully read a batch, yield the batch
            if batch_size and len(audio) == batch_size:
                yield audio_list, sampling_rates, filenames
                audio_list = []
                sampling_rates = []
                filenames = []

        yield audio_list, sampling_rates, filenames
