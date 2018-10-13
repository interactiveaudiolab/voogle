import argparse
import librosa
import os
import yaml
from flask import Flask, render_template, request
from werkzeug import secure_filename
from run_model_text import search_audio

app = Flask(__name__)


def str2bool(v):
    # parses various True/False command-line arguments
    # source: StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


@app.route("/")
def hello():
    filename_list = []
    f_list = os.listdir(app.config.get('database_directory'))
    for f in f_list:
        # grab all audio filenames in search database for autocomplete
        if ".wav" in f.lower() or ".mp3" in f.lower():
            filename_list.append(f[:-4])

    return render_template('index.html', filenamelist=filename_list)


@app.route('/load', methods=['POST'])
def load():
    # TODO: on startup, the code should load the model and database,
    # preprocess if necessary, and display loading status to user.
    # Meanwhile, the user should be able to provide their initial query,
    # which should be processed after loading is completed. A separate
    # loading indicator should be displayed to tell the user the progress
    # on their query.
    # Note: work on this after testing single-query case
    pass


@app.route('/search', methods=['POST'])
def search():

    # user's full audio recording
    file = request.files['file']

    # query time markers for trimming full audio
    offset = request.form['start']
    duration = request.form['length']

    # string containing text filter (optional)
    text = request.form['textDescription']

    if file:
        # write full audio recoding to disk
        recording_filename = secure_filename(file.filename) + '.wav'
        recording_filepath = os.path.join(
            app.config.get('query_path'), recording_filename)
        file.save(recording_filepath)

        # generate a new audio file based on query time markers
        query, sampling_rate = librosa.load(
            recording_filepath,
            sr=None,
            offset=float(offset),
            duration=float(duration))

        # write query to disk
        query_filepath = app.config.get('query_path') + '/query.wav'
        librosa.output.write_wav(query_filepath, query, sampling_rate)

        # run a similarity search between the query and the audio database
        sorted_filenames, sorted_filenames_matched = search_audio(
            query_filepath, text, app.config)

        # return the results to the frontend as a list with delimiter '...'
        results = sorted_filenames.tolist() + ['...'] + \
            sorted_filenames_matched.tolist()

        print(results)
    else:
        # TODO: send message back to upload file
        pass

    # can't send a list with Flask
    # make it a long string and deal with it on the front-end
    return ','.join(results)


if __name__ == "__main__":
    # set up parser to grab optional inputs:
    #   -c specifies the .yaml config file
    #   -d specifies debug mode on/off
    #   -t specifies threading on/off
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str,
        help="The filepath of the yaml config you wish to use.",
        default="./config/test.yaml")
    parser.add_argument(
        "-d", "--debug", type=str2bool,
        help="Sets debug=\"true\" if \"True\", false otherwise.",
        default=False)
    parser.add_argument(
        "-t", "--threaded", type=str2bool,
        help="Sets threaded=\"true\" if \"True\", false otherwise.",
        default=False)
    args = parser.parse_args()

    # Load the config file
    config = yaml.safe_load(open(args.config))

    # Setup the server process
    app.config.update(vars(args))
    app.config.update(config)
    app.run(debug=args.debug, threaded=args.threaded)
