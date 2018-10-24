import librosa
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from model.QueryByVoiceModel import QueryByVoiceModel


class SiameseStyle(QueryByVoiceModel):
    '''
    A siamese-style neural network for query-by-voice applications.

    citation: Y. Zhang, B. Pardo, and Z. Duan, "Siamese Style Convolutional
        Neural Networks for Sound Search by Vocal Imitation," in IEEE/ACM
        Transactions on Audio, Speech, and Language Processing, pp. 99-112,
        2018.
    '''

    def __init__(
        self, uses_windowing=False, window_length=4.0, hop_length=2.0):
        '''
        SiameseStyle model constructor.

        Arguments:
            uses_windowing: A boolean. Indicates whether the model slices the
                representation
            window_length: A float. The window length in seconds. Unused if
                uses_windowing is False.
            hop_length: A float. The hop length between windows in seconds.
                Unused if uses_windowing is False.
        '''
        super().__init__(uses_windowing, window_length, hop_length)

    def construct_representation(self, audio_list, sampling_rates, is_query):
        '''
        Constructs the audio representation used during inference. Audio
        files from the dataset are constructed only once and cached for
        later reuse.

        Arguments:
            audio_list: A python list of 1D numpy arrays. Each array represents
                one variable-length mono audio file.
            sampling_rate: A python list of ints. The corresponding sampling
                rate of each element of audio_list.
            is_query: A boolean. True only if audio is a user query.

        Returns:
            A python list of audio representations. The list order should be
                the same as in audio_list.
        '''

        # Siamese-style network requires different representation of query
        # and dataset audio
        if is_query:
            representation = self._construct_representation_query(
                audio_list[0], sampling_rates[0])
        else:
            representation = self._construct_representation_dataset(
                audio_list, sampling_rates)
        return representation

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
        self.logger.info('Loading model weights from {}'.format(model_filepath))
        filepath = os.path.join(os.path.dirname(__file__), model_filepath)
        self.model = load_model(filepath)
        self.graph = tf.get_default_graph()

    def predict(self, query, items):
        '''
        Runs model inference on the query.

        Arguments:
            query: An audio representation as defined by
                construct_representation. The user's vocal query.
            items: A python list of audio representations as defined by
                construct_representation. The dataset of potential matches for
                the user's query.

        Returns:
            A python list of floats. The similarity score of the query and each
                element in the dataset. The list order should be the same as
                in dataset.
        '''
        # add another dimension to each
        query = np.expand_dims(query, axis=1)
        items = np.expand_dims(np.array(items), axis=1)

        if not self.model:
            raise RuntimeError('No model loaded during call to predict.')

        # run model inference
        with self.graph.as_default():
            self.logger.debug('Running inference')
            return self.model.predict(
                [query, items], batch_size=1, verbose=1)

    def _construct_representation_query(self, query, sampling_rate):
        self.logger.debug('Constructing query representation')

        # resample query at 16k
        new_sampling_rate = 16000
        query = librosa.resample(query, sampling_rate, new_sampling_rate)
        sampling_rate = new_sampling_rate

        if self.uses_windowing:
            windows = self._window(query, sampling_rate)
        else:
            windows = [librosa.util.fix_length(query, 4 * sampling_rate)]

        # construct the logmelspectrogram of the signal
        representation = []
        for window in windows:
            melspec = librosa.feature.melspectrogram(
                window, sr=sampling_rate, n_fft=133,
                hop_length=133, power=2, n_mels=39,
                fmin=0.0, fmax=5000)
            melspec = melspec[:, :482]
            logmelspec = librosa.power_to_db(melspec, ref=np.max)

            # normalize to zero mean and unit variance
            representation.append(
                self._normalize(logmelspec).astype('float32'))

        return [np.array(representation)]

    def _construct_representation_dataset(self, dataset, sampling_rates):
        new_sampling_rate = 44100
        representations = []
        for audio, sampling_rate in zip(dataset, sampling_rates):

            # resample audio at 44.1k
            audio = librosa.resample(audio, sampling_rate, new_sampling_rate)
            sampling_rate = new_sampling_rate

            if self.uses_windowing:
                windows = self._window(audio, sampling_rate)
            else:
                windows = [librosa.util.fix_length(audio, 4 * sampling_rate)]

            representation = []
            for window in windows:
                # construct the logmelspectrogram of the signal
                melspec = librosa.feature.melspectrogram(
                    audio,
                    sr=sampling_rate,
                    n_fft=1024,
                    hop_length=1024,
                    power=2)
                melspec = melspec[:, 0:128]
                logmelspec = librosa.power_to_db(melspec, ref=np.max)

                # normalize to zero mean and unit variance
                representation.append(
                    self._normalize(logmelspec).astype('float32'))

            representations.append(np.array(representation))

        return representations

    def _normalize(self, x):
        # normalize to zero mean and unit variance
        mean = x.mean(keepdims=True)
        std = x.std(keepdims=True)
        return (x - mean) / std
