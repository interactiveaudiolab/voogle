import h5py
import librosa
import logging
import numpy as np
import os
from audioread import NoBackendError

def load_database(database_directory):
    # load all audio into memory
    # TODO: this will fail for large databases
    database = []
    sampling_rates = []
    filenames = []
    for filename in os.listdir(database_directory):
        try:
            # read the file in as an audio file
            filepath = os.path.join(database_directory, filename)
            # TODO: why does the query load at 16k and the database at 44.1k?
            datum, sampling_rate = librosa.load(filepath, sr=None)
            database.append(datum)
            sampling_rates.append(sampling_rate)
            filenames.append(filename)
        except NoBackendError:
            logging.warn('The file {} could not be decoded by any backend. \
                Either no backends are available or each available backend \
                failed to decode the file'.format(filename))
            continue

    return database, sampling_rates, filenames

def search_audio(query_filepath, model, config):
    # load query from disk and preprocess
    query, sampling_rate = librosa.load(query_filepath, sr=None)
    query = model.construct_representation([query], [sampling_rate], True)

    # determine the directory in which to store preprocessed data
    cache_directory = config.get('data_cache')
    cache_filename = config.get('model_name') + '-' + \
                     config.get('database_name')
    cache_filepath = os.path.join(cache_directory, cache_filename)

    try:
        # Attempt to load the cached database from disk
        # TODO: convert to either hdf5 or database
        # database = h5py.File(cache_filepath, 'r')
        database = np.load(cache_filepath + '_data.npy')
        filenames = np.load(cache_filepath + '_filenames.npy')
        logging.info('Found cached data for the {} model and the {} database'
            .format(config.get('model_name'), config.get('database_name')))
    except OSError as e:
        # create cache directory if it does not exist
        try:
            os.makedirs(cache_directory)
        except OSError as e:
            pass

        # failed to find cached data--must create from scratch
        database, sampling_rates, filenames = load_database(
            config.get('database_directory'))

        database = model.create_representation(database, sampling_rates, False)

        np.save(cache_filepath + '_data.npy', database)
        np.save(cache_filepath + '_filenames.npy', filenames)

    # prepare the model for inference
    model.load_model(config.get('model_filepath'))

    # run model inference and return the matching files in order of similarity
    return inference(database, filenames, query[0], model)

def inference(database, filenames, query, model):
    # Retrieve the similarity measure between query and each database entry
    model_output = np.array(model.predict(query, database))

    # Determine ranking
    sorted_index = np.argsort(model_output.flatten())[::-1]

    # Sort the files by similarity rank
    return filenames[sorted_index]