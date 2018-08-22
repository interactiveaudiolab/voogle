# vocalsearch
This is code for an audio search engine that uses vocal imitations of the desired sound

The microphone request for access, recording, and wav file generation is handled by code from Record.js, a JavaScript plugin for recording/exporting the output of Web Audio API nodes. It is licensed under the MIT license.

The plugin can be found here: https://github.com/mattdiamond/Recorderjs


## Use
**Note, you must be using Python 2.7. It is advisable to make a Virtual Environment in which to install the dependencies.**


Clone the repository, then navigate to /Recording/FlaskApp, then:
```
pip install -r requirements.txt
```
If this did not work, you may need to upgrade pip. To do so, please activate the virtual environment you created and:
```
curl https://bootstrap.pypa.io/get-pip.py | python
```
Then, please try the above pip install again.

After this, in order to host the app on your local machine, run: 
```
python app.py
```

This will start the web server. To access the app, please open your internet browser (Google Chrome preferred) and navigate to 
```
127.0.0.1:5000/
```
From there, please follow the directions on the website. Enjoy!
