from flask import Flask, render_template, request, redirect, url_for
from flask.json import jsonify
import os, keras, librosa, json, argparse
from werkzeug import secure_filename

from run_model_text import *


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT,'uploads')

app = Flask(__name__)

def str2bool(v): # code to accept multiple alternatives for True/False (from stackoverflow)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser() # set up parser to grab optional inputs: -m specifies model path, -d specifies debug mode on/off, -t specifies threading on/off, -db specifies sound database to search
parser.add_argument("-m", "--model", help="The filepath of the model you wish to use.", type=str, default="./model/model_11-10_top_pair.h5")
parser.add_argument("-d", "--debug", help="Sets debug=\"true\" if \"True\", false otherwise.", type=str2bool, default=False)
parser.add_argument("-t", "--threaded", help="Sets threaded=\"true\" if \"True\", false otherwise.", type=str2bool, default=False)
parser.add_argument("-db", "--database", help="The filepath of the database containing audio you wish to use as the search space", type=str, default="./static/")



args = parser.parse_args()

model_path = args.model
model = load_model(model_path)


@app.route("/")
def hello():
	filename_list = []
	f_list = os.listdir(args.database)
	for f in f_list:
		if ".wav" in f.lower() or ".mp3" in f.lower(): # grab all filenames of audio files in search database, for autocomplete
			filename_list.append(f[:-4]) 
	
	return render_template('index.html', filenamelist=filename_list)

@app.route('/search', methods = ['GET', 'POST'])
def search():
	
	# retrieve FormData from the AJAX request
	if request.method == 'POST':
		file = request.files['file'] # blob of audio file of the user recording
		offset = request.form['start'] # time offset of starting point of imitation (used to trim audio file)
		duration = request.form['length'] # time offset of ending point of imitation (used to trim audio file)
		text = request.form['textDescription'] # string containing text filter (optional)
		print text
		if file:
			# save incoming blob as .wav audio file
			filename = secure_filename(file.filename) + '.wav'
			file.save(os.path.join(UPLOAD_FOLDER, filename))

			# get its filepath now that it's been saved
			incoming_filepath = UPLOAD_FOLDER + '/' + filename

			#create a filepath for our soon-to-be edited version of the recording file
			outgoing_filepath = UPLOAD_FOLDER + '/query.wav'

			# generate a new audio file based on the start and endpoints from the wavesurfer region
			recording, sr = librosa.load(incoming_filepath, sr=None, offset=float(offset), duration=float(duration))
			
			# and save it to the filepath we created earlier
			librosa.output.write_wav(outgoing_filepath, recording, sr)

			# define filepaths for the keras model
			imi_path = outgoing_filepath
			ref_dir = './static/'

			# run the siamese network, giving it our recorded imitation, the directory of audio files it is to search, the path to the keras model, and the text query
			sorted_filenames, sorted_filenames_matched = search_audio(imi_path, ref_dir, model, text)

			#return the results to a Python list, delimit the matched vs unmatched lists with a string '...', this will be parsed on the front-end
			results = sorted_filenames.tolist() + ['...'] + sorted_filenames_matched.tolist()
			print results

	# can't send a list with Flask so we'll make it a long string and deal with it on the front-end
	return ','.join(results)

 
if __name__ == "__main__":
    app.run(debug=args.debug, threaded=args.threaded)