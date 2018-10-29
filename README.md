# Voogle
Voogle is an audio search engine that uses vocal imitations of the desired sound as the search query.

## Installation
### Server
Voogle backend dependencies are installed with `pip install -r requirements.txt`.

**Note:** Windows and Linux users must have [FFmpeg](https://www.ffmpeg.org/) installed.

### Interface
Voogle frontend dependencies are installed with `npm install`.

**Note:** You must have [Node.js](https://nodejs.org/en/) installed.

## Setup
After installing the dependencies, the Voogle app can be deployed.

### Deploying Locally
1. Start the server process by executing `npm run production`.
2. Navigate to `localhost:5000` in your browser.

From there, please follow the directions on the website. Enjoy!

## Testing
Unit tests can be run with `npm run test`.

## Extending
Voogle can be extended to incorporate additional models and datasets.

### Adding a model
- Define your model as a subclass of [`QueryByVoiceModel`](server/model/QueryByVoiceModel.py) with all abstract methods implemented as described.
- Add the model constructor to [`factory.py`](server/factory.py).
- Place your model's weights in [`server/model/weights/`](server/model/weights/).
- Update the model name and filepath in [`config.yaml`](server/config.yaml).

An example model can be found [here](server/model/SiameseStyle.py).

### Adding a dataset
- Define your dataset as a subclass of [`QueryByVoiceDataset`](server/data/QueryByVoiceDataset.py) with all abstract methods implemented as described.
- Add the dataset constructor to [`factory.py`](server/factory.py).
- Place the audio files in [`server/data/audio/<your_dataset_name>`](server/data/audio/).
- Update the dataset name in [`config.yaml`](server/config.yaml).
- If frontend audio retrieval is needed, the files must be hosted in the `voogle` S3 bucket. Contact maxrmorrison@gmail.com for more information.

An example dataset can be found [here](server/data/TestDataset.py).
