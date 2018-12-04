import argparse
import json
import librosa
import numpy as np
import os
import time
import yaml
from factory import dataset_factory, model_factory
from flask import Flask, jsonify, request, send_from_directory, send_file
from log import get_logger
from timeit import default_timer as timer
from voogle import Voogle
from werkzeug.exceptions import BadRequest

logger = get_logger('root')
app = Flask(__name__, static_url_path='', static_folder='')


@app.route('/')
def index():
    logger.debug('Rendering index.html')
    return send_from_directory('build', 'index.html')


@app.route('/retrieve', methods=['POST'])
def retrieve():
    logger.debug('Retrieved request for audio file')
    filename = request.form['filename']
    file = os.path.join(app.config.get('dataset_directory'), filename)
    try:
        return send_file(file)
    except FileNotFoundError:
        raise BadRequest('Audio file {} cannot be found'.format(filename))


@app.route('/search', methods=['POST'])
def search():
    start = timer()
    logger.debug('Retrieved search request')

    # fetch user's query
    query_file = request.files['query']
    sampling_rate = request.form['sampling_rate']
    text_input = request.form['text_input']

    if query_file:
        # Upack file stream and read bytes into numpy array
        query = np.frombuffer(query_file.read(), dtype=np.float32)
        logger.debug('Retrieved user query')

        # Decode sampling_rate from string
        try:
            sampling_rate = int(sampling_rate)
        except ValueError:
            logger.warning('Couldn\'t decode sampling rate. Attempting search\
                            with a sampling rate of 48000 Hz.')
            sampling_rate = 48000

        # write query to disk
        query_filepath = os.path.join(
            app.config.get('query_directory'),
            str(int(time.time())) + '_' + text_input + '.wav')
        save_start = timer()
        librosa.output.write_wav(query_filepath, query, sampling_rate)
        save_end = timer()

        logger.info('Saved query in {} seconds'.format(save_end - save_start))

        # run a similarity search between the query and the audio dataset
        voogle = app.config.get('voogle')
        ranked_matches, text_matches, similarity_scores = voogle.search(
            query, sampling_rate, text_input)
        logger.info('Produced matches {} with text-match array {}\
                    '.format(ranked_matches, text_matches))

        # Pass the results to the frontend
        logger.debug('Sending search request results to client')

        end = timer()
        logger.info(
            'Completed search request in {} seconds'.format(end - start))

        return jsonify({
            'matches': ranked_matches,
            'text_matches': text_matches,
            'similarity_scores': similarity_scores
        })
    else:
        # User did not provide a query
        logger.warning('A search was attempted with no query')
        return jsonify({'matches': [], 'text_matches': []})

if __name__ == '__main__':
    # set up parser to grab optional inputs:
    #   -c specifies the .yaml config file
    #   -d specifies debug mode on/off
    #   -t specifies threading on/off
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help='Run Flask with the debug flag enabled.',
        action='store_true')
    parser.add_argument(
        '-t', '--threaded',
        help='Run Flask with threading enabled.',
        action='store_true')
    args = parser.parse_args()

    # Load the config file
    parent_directory = os.path.dirname(__file__)
    config_file = os.path.join(parent_directory, 'config.yaml')
    config = yaml.safe_load(open(config_file))

    # Setup the model on the server
    model_filepath = os.path.join(
        parent_directory, 'model', config.get('model_filepath'))
    model = model_factory(
        config.get('model_name'), os.path.abspath(model_filepath))

    # Setup the dataset
    dataset_directory = os.path.join(
        parent_directory, 'data', 'audio', config.get('dataset_name'))
    representation_directory = os.path.join(
        parent_directory,
        'data',
        config.get('representation_directory'),
        config.get('dataset_name'),
        config.get('model_name'))
    dataset = dataset_factory(
        config.get('dataset_name'),
        dataset_directory,
        representation_directory,
        config.get('measure_similarity_batch_size'),
        config.get('construct_representation_batch_size'),
        model)

    voogle = Voogle(model, dataset, config.get('require_text_match'))

    query_directory = os.path.join(
        parent_directory, 'data', 'weights', config.get('query_path'))

    # Make the query directory if it doesn't exist
    try:
        os.makedirs(query_directory)
    except OSError:
        pass

    app.config.update(config)
    app.config.update({'voogle': voogle})
    app.config.update({'dataset_directory': dataset_directory})
    app.config.update({'query_directory': query_directory})
    app.run(debug=args.debug, threaded=args.threaded)
