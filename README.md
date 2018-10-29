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
- Define your model as a subclass of `QueryByVoiceModel` with all abstract methods implemented as described in the base class.
- Add the model constructor to `factory.py`.
- Place your model's weights in `server/model/weights/`.
- Update the model name and filepath in `config.yaml`.
An example model can be found

### Adding a dataset
- Define your dataset as a subclass of `QueryByVoiceDataset` with all abstract methods implemented as described in the base class.
- Add the dataset constructor to `factory.py`.
- Place the audio files in `server/data/audio/<your_dataset_name>`.
- Update the dataset name in `config.yaml`.
An example dataset can be found
