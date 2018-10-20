import argparse
import librosa
import logging
import os
import yaml
from flask import Flask, request, send_from_directory
from factory import dataset_factory, model_factory
from VocalSearch import VocalSearch

logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf'))
logger = logging.getLogger('root')

app = Flask(__name__, static_folder='../build')


@app.route('/')
def index():
    logger.debug('Rendering index.html from root')
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/search', methods=['POST'])
def search():
    logger.debug('Retrieved search request')

    # fetch user's query
    query_file = request.files['query']
    sampling_rate = request.form['sampling_rate']
    offset = request.form['start']
    duration = request.form['length']

    if query_file:
        logger.debug('Retrieved user query')

        # write query to disk
        query_filepath = app.config.get('query_path') + '/query.wav'
        query_file.save(query_filepath)

        query, sampling_rate = librosa.load(
            query_filepath,
            sr=None,
            offset=float(offset),
            duration=float(duration))

        # run a similarity search between the query and the audio dataset
        vocal_search = app.config.get('vocal_search')
        ranked_matches = vocal_search.search(query, sampling_rate)
        logger.info(ranked_matches)

        # Pass the results to the frontend
        return ','.join(ranked_matches)
    else:
        # User did not provide a query
        logger.warning('A search was attempted with no query')
        return ''


if __name__ == '__main__':
    # set up parser to grab optional inputs:
    #   -c specifies the .yaml config file
    #   -d specifies debug mode on/off
    #   -t specifies threading on/off
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str,
        help='The filepath of the yaml config you wish to use.',
        default='./server/config/test.yaml')
    parser.add_argument(
        '-d', '--debug',
        help='Run Flask with the debug flag enabled.',
        action='store_true')
    parser.add_argument(
        '-t', '--threaded',
        help='Run Flask with threading enabled.',
        action='store_true')
    args = parser.parse_args()

    # # Load the config file
    config = yaml.safe_load(open(args.config))

    # # Setup the model on the server
    model = model_factory(
        config.get('model_name'), config.get('model_filepath'))

    # # Get a generator for the audio representations
    dataset = dataset_factory(
        config.get('dataset_name'),
        config.get('dataset_directory'),
        config.get('representation_directory'),
        config.get('similarity_model_batch_size'),
        config.get('representation_batch_size'),
        model)

    vocal_search = VocalSearch(model, dataset)

    app.config.update(config)
    app.config.update({'vocal_search': vocal_search})
    app.run(debug=args.debug, threaded=args.threaded)
