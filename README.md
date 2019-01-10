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

## Available Datasets
Any collection of audio files can be used as sounds returned by Voogle in response to a vocal query. The Interactive Audio Lab has released 2 datasets specifically for the training of query-by-vocal-imitation models: [Vocal Imitation Set](https://zenodo.org/record/1340763#.XAap0mhKiM8) and [VocalSketch](https://zenodo.org/record/1251982#.XAap1WhKiM8) [1, 2]. A small test dataset for demos can be downloaded [here](https://www.dropbox.com/s/lkj55uvz4z26i8d/test_dataset.zip?dl=1).

Audio files should be placed in [`data/audio/<dataset_name>`](data/audio/). The dataset used during execution can be specified in [`config.yaml`](config.yaml).

## Available Models
Interactive Audio Lab has released the following models for query-by-vocal-imitation:
 - `siamese-style`: a siamese-style neural network [3]
    - [weight file](https://www.dropbox.com/s/234i2ft9sfcdpty/siamese_style.h5?dl=1)
 - `VGGish-embedding`: cosine similarity of VGGish embeddings
    - [weight file](https://www.dropbox.com/s/5x5ceczislmyk0y/vggish_pretrained_convs.pth?dl=1)

Weight files should be placed in [`model/weights`](model/weights/). The model used during execution can be specified in [`config.yaml`](config.yaml).

## Setup
After installing the dependencies, a dataset, and a model, the Voogle app can be deployed.

### Deploying Locally
1. Start the server by running `npm run production`.
2. Navigate to `localhost:5000` in your browser.

From there, please follow the directions on the website. Enjoy!

**Note:** There are currently two frontend interfaces available for Voogle. If you would like to use the alternate interface, use the command `npm run old-interface` instead during step 1.

### Deploying on the Web
The following steps are required to prepare Voogle for deployment as a web application:
- [ ] Get a domain name
- [ ] Determine server location (e.g., AWS)
- [ ] Setup server
- [ ] Serve audio files outside of the main event loop (e.g., with Apache or nginx)
- [ ] Update query file naming. Depending on the scale, multiple users could send a query at the same time. This would require query names to include a unique id that is not exclusively associated with the timestamp.
- [ ] Test concurrency with multiple simultaneous users

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

## References
- [1] Bongjun Kim, Madhav Ghei, Bryan Pardo, and Zhiyao Duan, "Vocal Imitation Set: a dataset of vocally imitated sound events using the AudioSet ontology," Proceedings of the Detection and Classification of Acoustic Scenes and Events 2018 Workshop (DCASE2018), Surrey, UK, Nov. 2018. [[paper link](http://dcase.community/documents/workshop2018/proceedings/DCASE2018Workshop_Kim_135.pdf)]
- [2] Mark Cartwright and Bryan Pardo, "Vocalsketch: Vocally imitating audio concepts," Proceedings of the 33rd Annual ACM Conference on Human Factors in Computing Systems (ACM), 2015. [[paper link](http://music.cs.northwestern.edu/publications/cartwright_pardo_chi2015.pdf)]
- [3] Yichi Zhang, Bryan Pardo, and Zhiyao Duan, "Siamese Style Convolutional Neural Networks for Sound Search by Vocal Imitation," IEEE/ACM Transactions on Audio Speech and Language Processing. [[paper link](https://ieeexplore.ieee.org/document/8453811)]
