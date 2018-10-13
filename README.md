# vocalsearch
VocalSearch is an audio search engine that uses vocal imitations of the desired sound as the search query.

```
pip install -r requirements.txt
```

After this, in order to host the app on your local machine, run: 
```
python app.py
```
<!-- TODO: edit -->
There are also some optional arguments you can pass in to the app.py program. 
* `-m` lets you specify a filepath to a model so you can hotswap models (default is `./model/model_11-10_top_pair.h5`)
* `-d` lets you specify whether the debug option in Flask is to be used (default is False)
* `-t` lets you specify whether the threading option in Flask should be used (default is False)
* `-db` lets you specify a directory you would like to use for audio files (default is `./static/`)... this is not working for search just yet, but it does work when generating the list of suggestions for autocomplete. Please see the issue on this in the Issues section.

This will start the web server. To access the app, please open your internet browser (Google Chrome preferred) and navigate to 
```
127.0.0.1:5000/
```
From there, please follow the directions on the website. Enjoy!

## Note 
The microphone request for access, recording, and wav file generation is handled by code from Record.js, a JavaScript plugin for recording/exporting the output of Web Audio API nodes. It is licensed under the MIT license.

The plugin can be found here: https://github.com/mattdiamond/Recorderjs
