import librosa
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
from model.QueryByVoiceModel import QueryByVoiceModel
from model.vggish_utils import vggish_input_bk
from model.vggish_utils.vggish_model_architecture import VGGish2s
from scipy import spatial
import torch
from torch.autograd import Variable


class VGGishEmbedding(QueryByVoiceModel):
    '''
    A VGGish model to extract feature embeddings for query-by-voice applications.

    citation: S.Hershey,S.Chaudhuri,D.P.Ellis,J.F.Gemmeke,A.Jansen, R. C. Moore,
    M. Plakal, D. Platt, R. A. Saurous, B. Seybold, et al.,
    “Cnn architectures for large-scale audio classification,”
    in Acoustics, Speech and Signal Processing (ICASSP),
    2017 IEEE International Conference on. IEEE, 2017, pp. 131–135.
    '''

    def __init__(
        self,
        model_filepath,
        parametric_representation=False,
        uses_windowing=False,
        window_length=None,
        hop_length=None):
        '''
        SiameseStyle model constructor.

        Arguments:
            model_filepath: A string. The path to the model weight file on
                disk.
            parametric_representation: A boolen. True if the audio
                representations depend on the model weights.
            uses_windowing: A boolean. Indicates whether the model slices the
                representation
            window_length: A float. The window length in seconds. Unused if
                uses_windowing is False.
            hop_length: A float. The hop length between windows in seconds.
                Unused if uses_windowing is False.
        '''
        super().__init__(
            model_filepath,
            parametric_representation,
            uses_windowing,
            window_length,
            hop_length)

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
        pairs = zip(audio_list, sampling_rates)
        return [self._construct_representation(a, s) for (a, s) in pairs]

    def measure_similarity(self, query, items):
        '''
        Runs model inference on the query.

        Arguments:
            query: A numpy array. An audio representation as defined by
                construct_representation. The user's vocal query.
            items: A numpy array. The audio representations as defined by
                construct_representation. The dataset of potential matches for
                the user's query.

        Returns:
            A python list of floats. The similarity score of the query and each
                element in the dataset. The list order should be the same as
                in dataset.
        '''
        if not self.model:
            raise RuntimeError('No model loaded during call to predict.')

        # run model inference
        self.logger.debug('Running inference')
        simlarities = []
        for q, i in zip(query, items):
            simlarities.append(1 - spatial.distance.cosine(q, i))

        return np.array(simlarities)

    def _load_model(self):
        '''
        Loads the model weights from disk. Prepares the model to be able to
        make predictions.
        '''
        self.logger.info(
            'Loading model weights from {}'.format(self.model_filepath))
        self.model = VGGish2s()
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.model.eval()

    def _construct_representation(self, audio, sampling_rate):
        # resample query at 16k
        new_sampling_rate = 16000
        audio = librosa.resample(audio, sampling_rate, new_sampling_rate)
        sampling_rate = new_sampling_rate

        # zero-padding
        target_length = int(np.ceil(audio.shape[0]/sampling_rate))
        if target_length % 2 != 0:
            target_length += 1
        pad = np.zeros((target_length*sampling_rate-audio.shape[0]))
        audio = np.append(audio, pad)

        melspec = vggish_input_bk.waveform_to_examples(audio, sampling_rate)
        melspec = melspec.astype('float32')
        representation = self.model(Variable(torch.from_numpy(melspec)))
        representation = representation.detach().numpy()

        return representation
