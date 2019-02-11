import os
from model.MCFT import MCFT
from model.SiameseStyle import SiameseStyle
from model.VGGishEmbedding import VGGishEmbedding
from data.TestDataset import TestDataset
from data.OtoMobile import OtoMobile
from log import get_logger

logger = get_logger('factory')


def model_factory(model_name, model_filepath):
    '''
    Given a model name and weight file location, construct the model for
    query-by-voice search.

    Arguments:
        model_name: A string. The name of the model.
        model_filepath: A string. The location of the weight file on disk.

    Returns:
        A QueryByVoiceModel.
    '''
    logger.debug('Attempting to load the {} model from {}'.format(
        model_name, model_filepath))

    if model_name == 'mcft':
        model = MCFT(model_filepath)
    elif model_name == 'siamese-style':
        model = SiameseStyle(model_filepath)
    elif model_name == 'VGGish-embedding':
        model = VGGishEmbedding(model_filepath)
    else:
        raise ValueError('Model {} is not defined'.format(model_name))

    logger.debug('Model loading complete')
    return model


def dataset_factory(
    dataset_name,
    dataset_directory,
    representation_directory,
    construct_representation_batch_size,
    measure_similarity_batch_size,
    model):
    '''
    Constructs a dataset object for query-by-voice search.

    Arguments:
        dataset_name: A string. The name of the dataset.
        dataset_directory: A string. The location of the audio files.
        representation_directory: A string. The location of the corresponding
            audio representations.
        construct_representation_batch_size: An integer or None. The maximum
            number of audio files to load during one batch of representation
            construction.
        measure_similarity_batch_size: An integer or None. The maximum number of
            representations to load during one batch of model inference.
        model: A QueryByVoiceModel. The model being used in the query-by-voice
            system. Defines the audio representation.

    Returns:
        A Dataset object.
    '''
    logger.debug('Attempting to construct the {} dataset in {}. \
        Representations will be stored in {}'.format(
            dataset_name, dataset_directory, representation_directory))

    if dataset_name == 'test_dataset':
        dataset = TestDataset(
            dataset_directory,
            representation_directory,
            model,
            measure_similarity_batch_size,
            construct_representation_batch_size)
    elif dataset_name == 'otomobile':
        dataset = OtoMobile(
            dataset_directory,
            representation_directory,
            model,
            measure_similarity_batch_size,
            construct_representation_batch_size)
    else:
        raise ValueError('Dataset {} is not defined'.format(dataset_name))

    logger.debug('Dataset construction complete.')

    return dataset
