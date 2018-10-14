from SiameseStyle import SiameseStyle
from TestDataset import TestDataset


def model_factory(model_name, model_filepath):
    '''
    TODO
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
    TODO
    '''
    if dataset_name == 'test_dataset':
        dataset = TestDataset(dataset_directory, representation_directory)
    else:
        raise ValueError('Dataset {} is not defined'.format(dataset_name))

    return dataset.data_generator(
        model, similarity_model_batch_size, representation_batch_size)
