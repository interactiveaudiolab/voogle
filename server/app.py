# TODO pass filenames to frontend
# TODO prevent duplicate log file entries
# TODO regenerate representation on wav file update
# TODO get pylint working again
import argparse
import librosa
import logging
import os
import yaml
from flask import Flask, request, send_from_directory
from flask import Flask
from factory import dataset_factory, model_factory
from VocalSearch import VocalSearch

logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf'))
logger = logging.getLogger('root')

# TODO: If we want concurrent user access, we will need something more powerful
# than our current Flask setup
app = Flask(__name__, static_folder='../build')


@app.route('/')
def index():
    logger.debug('Rendering index.html from root')
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/search', methods=['POST'])
def search():
    logger.debug('Retrieved search request')

    # fetch user's query
    query = request.files['query']
    sampling_rate = request.form['sampling_rate']

    if query:
        # write query to disk
        query_filepath = app.config.get('query_path') + '/query.wav'
        librosa.output.write_wav(query_filepath, query, sampling_rate)

        # run a similarity search between the query and the audio dataset
        vocal_search = app.config.get('vocal_search')
        ranked_matches = vocal_search.search_audio(
            query, sampling_rate, model, dataset)
        logger.info(ranked_matches)

        # Pass the results to the frontend
        return ','.join(ranked_matches)
    else:
        # User did not provide a query
        # TODO: send message back to upload query
        pass


def str2bool(v):
    '''
    Parses various True/False command-line arguments
    source: StackOverflow
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
        '-d', '--debug', type=str2bool,
        help='Sets debug=\'true\' if \'True\', false otherwise.',
        default=False)
    parser.add_argument(
        '-t', '--threaded', type=str2bool,
        help='Sets threaded=\'true\' if \'True\', false otherwise.',
        default=False)
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
