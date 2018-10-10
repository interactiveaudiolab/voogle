import os, librosa, argparse
from flask import Flask, render_template, request
from werkzeug import secure_filename
from run_model_text import search_audio

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT,'uploads')

def str2bool(v): 
    # accepts alternatives for True/False command arguments
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
    f_list = os.listdir(args.database)
    for f in f_list:
        # grab all filenames of audio files in search database, for autocomplete
        if ".wav" in f.lower() or ".mp3" in f.lower(): 
            filename_list.append(f[:-4]) 
    
    return render_template('index.html', filenamelist=filename_list)

@app.route('/search', methods = ['POST'])
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
        filename_full = secure_filename(file.filename) + '.wav'
        filepath_full = os.path.join(UPLOAD_FOLDER, filename_full)
        file.save(filepath_full)

        # generate a new audio file based on query time markers
        query, sampling_rate = librosa.load(
            filepath_full,
            sr=None,
            offset=float(offset),
            duration=float(duration))
        
        # write query to disk
        filepath_query = UPLOAD_FOLDER + '/query.wav'
        librosa.output.write_wav(filepath_query, recording, sampling_rate)

        # define filepaths for the keras model
        ref_dir = './static/'

        # run a similarity search between the query and the audio database
        sorted_filenames, sorted_filenames_matched = search_audio(
            filepath_query,
            app.config.get['database'],
            app.config.get['model'],
            text)

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
    #   -m specifies model path
    #   -d specifies debug mode on/off
    #   -t specifies threading on/off
    #   -db specifies sound database to search
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "-m", "--model", type=str,
        help="The filepath of the model you wish to use.",
        default="./model/model_11-10_top_pair.h5")
    parser.add_argument(
        "-d", "--debug", type=str2bool,
        help="Sets debug=\"true\" if \"True\", false otherwise.",
        default=False)
    parser.add_argument(
        "-t", "--threaded", type=str2bool, 
        help="Sets threaded=\"true\" if \"True\", false otherwise.",
        default=False)
    parser.add_argument("-db", "--database", type=str,
        help="The filepath of the database containing audio you wish to \
        use as the search space",
        default="./static/")
    args = parser.parse_args()

    # Load the model from disk
    model_path = args.model
    model = load_model(model_path)

    # Setup the server process
    app.config.update(vars(args))
    app.config['model'] = model
    app.run(debug=args.debug, threaded=args.threaded)