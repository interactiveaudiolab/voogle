# Voogle
Voogle is an audio search engine that uses vocal imitations of the desired sound as the search query.

Voogle is built in Python 3.6 and Javascript, using Node.js.

## Installation
### Server
Voogle backend dependencies are installed with `pip install -r requirements.txt`.

**Note:** Windows and Linux users must have [FFmpeg](https://www.ffmpeg.org/) installed.

### Interface
Voogle frontend dependencies are installed with `npm install`.

**Note:** You must have [Node.js](https://nodejs.org/en/) installed before you can run `npm install`.

## Setup
After installing the dependencies, the Voogle app can be deployed.

### Deploying Locally
1. Start the server by running `npm run production`.
2. Navigate to `localhost:5000` in your browser.

From there, please follow the directions on the website. Enjoy!

## Available Datasets
Any collection of audio files can be used as the sounds returned by Voogle in response to a vocal query. The Interactive Audio Lab has released 2 datasets specifically for the training of query-by-vocal-imitation models: [Vocal Imitation Set](https://zenodo.org/record/1340763#.XAap0mhKiM8) and [VocalSketch](https://zenodo.org/record/1251982#.XAap1WhKiM8). A small test dataset for demos can be downloaded [here](https://www.dropbox.com/s/lkj55uvz4z26i8d/test_dataset.zip?dl=1).

Audio files should be placed in [`data/audio/<dataset_name>`](data/audio/). The dataset used during exection can be specified in [`config.yaml`](config.yaml).

## Available Models
Interactive Audio Lab has released the following models for query-by-vocal-imitation:
 - `siamese-style`: a siamese-style neural network
    - [paper link](https://www.researchgate.net/publication/327407400_Siamese_Style_Convolutional_Neural_Networks_for_Sound_Search_by_Vocal_Imitation)
    - [weight file](https://www.dropbox.com/s/234i2ft9sfcdpty/siamese_style.h5?dl=1)
 - `VGGish-embedding`: cosine similarity of VGGish embeddings
    - [weight file](https://www.dropbox.com/s/5x5ceczislmyk0y/vggish_pretrained_convs.pth?dl=1)

Weight files should be placed in [`model/weights`](model/weights/). The model used during execution can be specified in [`config.yaml`](config.yaml).

## Testing
Unit tests can be run with `npm run test`.

## Extending
Voogle can be extended to incorporate additional models and datasets. If you would like to make your model or dataset available to all users of Voogle, contact interactiveaudiolab@gmail.com.

### Adding a model
- Define your model as a subclass of [`QueryByVoiceModel`](model/QueryByVoiceModel.py) with all abstract methods implemented as described.
- Add the model constructor to [`factory.py`](factory.py).
- Place your model's weights in [`model/weights`](model/weights/).
- Update the model name and filepath in [`config.yaml`](config.yaml).

An example model can be found [here](model/SiameseStyle.py).

### Adding a dataset
- Define your dataset as a subclass of [`QueryByVoiceDataset`](data/QueryByVoiceDataset.py) with all abstract methods implemented as described.
- Add the dataset constructor to [`factory.py`](factory.py).
- Place the audio files in [`data/audio/<dataset_name>`](data/audio/).
- Update the dataset name in [`config.yaml`](config.yaml).

An example dataset can be found [here](data/TestDataset.py).
