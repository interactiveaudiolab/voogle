import argparse
import librosa
import logging
import logging.config
import os
import yaml

# Setup logging
logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf'))
logger = logging.getLogger('root')

from flask import Flask, jsonify, request, send_from_directory
from factory import dataset_factory, model_factory
from Voogle import Voogle

app = Flask(__name__, static_folder='../build')


@app.route('/')
def index():
    logger.debug('Rendering index.html')
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/search', methods=['POST'])
def search():
    logger.debug('Retrieved search request')

    # fetch user's query
    query_file = request.files['query']
    sampling_rate = request.form['sampling_rate']
    offset = request.form['start']
    duration = request.form['length']
    text_input = request.form['text_input']

    if query_file:
        logger.debug('Retrieved user query')

        # write query to disk
        query_filepath = app.config.get('query_directory') + '/query.wav'
        query_file.save(query_filepath)

        query, sampling_rate = librosa.load(
            query_filepath,
            sr=None,
            offset=float(offset),
            duration=float(duration))

        # run a similarity search between the query and the audio dataset
        voogle = app.config.get('voogle')
        ranked_matches, text_matches = voogle.search(
            query, sampling_rate, text_input)
        logger.info('Produced matches {} with text-match array {}\
                    '.format(ranked_matches, text_matches))

        # Prepend the file location for S3 retreival
        bucket_directory = 'audio/' + app.config.get('dataset_name') + '/'
        ranked_matches = [bucket_directory + match for match in ranked_matches]

        # Pass the results to the frontend
        logger.debug('Sending search request results to client')
        return jsonify({
            'matches': ranked_matches,
            'text_matches': text_matches
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
        parent_directory,
        'data',
        config.get('dataset_directory'),
        config.get('dataset_name'))
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
        config.get('similarity_model_batch_size'),
        config.get('representation_batch_size'),
        model)

    voogle = Voogle(model, dataset)

    query_directory = os.path.join(
        parent_directory, 'data', config.get('query_path'))

    app.config.update(config)
    app.config.update({'voogle': voogle})
    app.config.update({'query_directory': query_directory})
    app.run(debug=args.debug, threaded=args.threaded)
