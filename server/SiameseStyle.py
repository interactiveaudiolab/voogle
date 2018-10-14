import librosa
import numpy as np
from keras.models import load_model
from QueryByVoiceModel import QueryByVoiceModel


class SiameseStyle(QueryByVoiceModel):
    '''
    TODO: verify with Madhav that default weights do correspond with trained
        siamese model

    A siamese-style neural network for query-by-voice applications.

    citation: Y. Zhang, B. Pardo, and Z. Duan, "Siamese Style Convolutional
        Neural Networks for Sound Search by Vocal Imitation," in IEEE/ACM
        Transactions on Audio, Speech, and Language Processing, pp. 99-112,
        2018.
    '''

    def __init__(self):
        super().__init__()

    def get_name(self):
        '''
        Get the model name.

        Returns:
            A string.
        '''
        return 'siamese-style'

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
            return self._construct_representation_query(
                audio_list[0], sampling_rates[0])
        else:
            return self._construct_representation_dataset(
                audio_list, sampling_rates)

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
        self.model = load_model(model_filepath)

    def predict(self, query, dataset):
        '''
        Runs model inference on the query.

        Arguments:
            query: An audio representation as defined by
                construct_representation. The user's vocal query.
            dataset: A python list of audio representations as defined by
                construct_representation. The dataset of potential matches for
                the user's query.

        Returns:
            A python list of floats. The similarity score of the query and each
                element in the dataset. The list order should be the same as
                in dataset.
        '''

        # replicate the query to pair against each dataset entry
        query = np.repeat(np.array(query), len(dataset), axis=0)

        # add another dimension to each
        # TODO: why?
        query = np.expand_dims(query, axis=1)
        dataset = np.expand_dims(np.array(dataset), axis=1)

        if not self.model:
            raise RuntimeError('No model loaded during call to predict.')

        # run model inference
        return self.model.predict(
            [query, dataset],  batch_size=1, verbose=1)

    def _construct_representation_query(self, query, sampling_rate):
        # resample query at 16k
        new_sampling_rate = 16000
        query = librosa.resample(query, sampling_rate, new_sampling_rate)
        sampling_rate = new_sampling_rate

        # force all queries to be 4-seconds long
        query = librosa.util.fix_length(query, 4 * sampling_rate)

        # construct the logmelspectrogram of the signal
        melspec = librosa.feature.melspectrogram(
            query, sr=sampling_rate, n_fft=133,
            hop_length=133, power=2, n_mels=39,
            fmin=0.0, fmax=5000)
        melspec = melspec[:, :482]
        logmelspec = librosa.power_to_db(melspec, ref=np.max)

        # normalize to zero mean and unit variance
        return [self.normalize(logmelspec).astype('float32')]

    def _construct_representation_dataset(self, dataset, sampling_rates):
        new_sampling_rate = 44100
        spectrograms = []
        for audio, sampling_rate in zip(dataset, sampling_rates):

            # resample audio at 44.1k
            audio = librosa.resample(audio, sampling_rate, new_sampling_rate)
            sampling_rate = new_sampling_rate

            # force audio to be 4-seconds long at 44.1k
            audio = librosa.util.fix_length(audio, 4 * sampling_rate)

            # construct the logmelspectrogram of the signal
            melspec = librosa.feature.melspectrogram(
                audio, sr=sampling_rate, n_fft=1024, hop_length=1024, power=2)
            melspec = melspec[:, 0:128]
            logmelspec = librosa.power_to_db(melspec, ref=np.max)

            # normalize to zero mean and unit variance
            normed = self.normalize(logmelspec).astype('float32')
            spectrograms.append(normed)

        return spectrograms

    def _normalize(x):
        # normalize to zero mean and unit variance
        mean = x.mean(keepdims=True)
        std = x.std(keepdims=True)
        return (x - mean) / std
