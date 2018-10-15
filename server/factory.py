from SiameseStyle import SiameseStyle
from TestDataset import TestDataset


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
    if model_name == 'siamese-style':
        model = SiameseStyle()
    else:
        raise ValueError('Model {} is not defined'.format(model_name))

    model.load_model(model_filepath)
    return model


def dataset_factory(dataset_name, dataset_directory, representation_directory,
                    representation_batch_size, similarity_model_batch_size,
                    model):
    '''
    Constructs a dataset object for query-by-voice search.

    Arguments:
        dataset_name: A string. The name of the dataset.
        dataset_directory: A string. The location of the audio files.
        representation_directory: A string. The location of the corresponding
            audio representations.
        representation_batch_size: An integer or None. The maximum number of
            audio files to load during one batch of representation
            construction.
        similarity_model_batch_size: An integer or None. The maximum number of
            representations to load during one batch of model inference.
        model: A QueryByVoiceModel. The model being used in the query-by-voice
            system. Defines the audio representation.

    Returns:
        A python generator used to generate representations.
    '''
    if dataset_name == 'test_dataset':
        dataset = TestDataset(dataset_directory, representation_directory)
    else:
        raise ValueError('Dataset {} is not defined'.format(dataset_name))

    return dataset.data_generator(
        model, similarity_model_batch_size, representation_batch_size)
