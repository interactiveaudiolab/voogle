import h5py
import librosa
import logging
import numpy as np
import os
from audioread import NoBackendError
from keras.models import load_model

def normalize(x):
    # normalize to zero mean and unit variance
    mean = x.mean(keepdims=True)
    std = x.std(keepdims=True)
    return (x - mean) / std

def preprocess_database(database_directory):

    # get a list of audio files
    all_files = os.listdir(database_directory)

    # load all audio into memory
    # TODO: this will fail for large databases
    spectrograms = []
    filenames = []
    for filename in all_files:

        try:
            # read the file in as an audio file
            filepath = os.path.join(database_directory, filename)
            # TODO: why does the query load at 16k and the database at 44.1k?
            audio, sampling_rate = librosa.load(filepath, sr=44100)
        except NoBackendError:
            logging.warn('The file {} could not be decoded by any backend. \
                Either no backends are available or each available backend \
                failed to decode the file'.format(filename))
            continue

        # force audio to be 4-seconds long at 44.1k
        audio = librosa.util.fix_length(audio, 4 * sampling_rate)

        # construct the logmelspectrogram of the signal
        # TODO: reduce code duplication
        melspec = librosa.feature.melspectrogram(
            y=audio, sr=sampling_rate, n_fft=1024, hop_length=1024, power=2)
        melspec = melspec[:, 0:128]
        logmelspec = librosa.power_to_db(melspec, ref=np.max)

        # normalize to zero mean and unit variance
        normed = normalize(logmelspec)

        spectrograms.append(normed)
        filenames.append(filename)

    spectrograms = np.array(spectrograms).astype('float32')

    return np.array(filenames), spectrograms

def audio_preprocessing(audio_filepath):

    # load audio
    audio, sampling_rate = librosa.load(audio_filepath, sr=16000)

    # force all audio queries to be 4-seconds long at 16k
    audio = librosa.util.fix_length(audio, 4 * sampling_rate)

    # construct the logmelspectrogram of the signal
    melspec = librosa.feature.melspectrogram(
        y=audio, sr=sampling_rate, n_fft=133,
        hop_length=133, power=2, n_mels=39, 
        fmin=0.0, fmax=5000)
    melspec = melspec[:, :482]
    logmelspec = librosa.power_to_db(melspec, ref=np.max)

    # normalize to zero mean and unit variance
    normed = normalize(logmelspec)

    # add batch dimension for network compatibility
    return np.expand_dims(normed, axis=0).astype('float32')

def search_audio(query_filepath, text, config):

    # TODO: the model must specify the preprocessing
    #   so the order should be load model -> load query ->
    #   preproc query -> load database
    # load query from disk and preprocess
    query = audio_preprocessing(query_filepath)

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
        database, filenames = preprocess_database(
            config.get('database_directory'))
        np.save(cache_filepath + '_data.npy', database)
        np.save(cache_filepath + '_filenames.npy', filenames)

    # Load the keras model from a hdf5 file
    model = load_model(config.get('model_filepath'))

    # run model inference and return the matching files in order of similarity
    return inference(database, filenames, query, model)

def inference(database, filenames, query, model):

    # replicate the query to pair against each database entry
    query = np.repeat(query, len(database), axis=0)

    # add another dimension to each
    # TODO: why?
    query = np.expand_dims(query, axis=1)
    databse = np.expand_dims(database, axis=1)

    # run model inference
    model_output = model.predict([query, database],  batch_size=1, verbose=1)
    model_output = np.array(model_output)
    sorted_index = np.argsort(model_output.flatten())[::-1]

    # sort the files by similarity rank
    return filenames[sorted_index]