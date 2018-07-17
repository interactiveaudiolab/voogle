from flask import Flask, render_template, request
from flask.json import jsonify
import os, keras, librosa, json
from werkzeug import secure_filename

from run_model_pytorch import *

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT,'uploads')

app = Flask(__name__)
 
@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/search', methods = ['GET', 'POST'])
def search():
	if request.method == 'POST':
		file = request.files['file']
		offset = request.form['start']
		duration = request.form['length']
		text = request.form['textDescription']
		print text
		if file:
			# save incoming audio file
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
			model_path = './model/random_selection'

			# run the siamese network
			sorted_filenames = search_audio(imi_path, ref_dir, model_path)

			#return the results to a Python list
			results = sorted_filenames.tolist()
			print results

	# can't send a list with Flask so we'll make it a long string and parse on the front end
	return ','.join(results)

 
if __name__ == "__main__":
    app.run(debug=True)