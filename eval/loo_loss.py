import librosa
import numpy as np
import os
import yaml
from factory import dataset_factory, model_factory
from voogle import Voogle


def compute_loss(voogle, dataset):
    losses = []
    for file in dataset._get_audio_filenames():
        filename = os.path.join(dataset.dataset_directory, file)
        audio, sampling_rate = librosa.load(filename, sr=None)
        _, filenames, _, sim = (voogle.search(audio, sampling_rate))
        scores = [sim[i] for i in range(len(sim)) if filenames[i] != filename]
        filenames = [f for f in filenames if f != filename]
        labels = [f.split('\\')[1] for f in filenames]
        unique_labels = set(labels)
        label_sim = {}
        for label in unique_labels:
            label_scores = [
                scores[i] for i in range(len(scores)) if labels[i] == label]
            label_sim[label] = np.mean(label_scores)
        ground_truth = filename.split('\\')[-2]
        pred = max(label_sim, key=label_sim.get)
        loss = label_sim[ground_truth] == max(label_sim.values())
        losses.append(loss)
        print('Ground truth: {}, Prediction: {}'.format(ground_truth, pred))

    return np.mean(np.array(losses))

if __name__ == '__main__':
    parent_directory = os.path.dirname(os.path.dirname(__file__))
    config_file = os.path.join(parent_directory, 'config.yaml')
    config = yaml.safe_load(open(config_file))

    # Setup the model
    model_filepath = os.path.join(
        parent_directory, 'model', 'weights', config.get('model_filepath'))
    model = model_factory(
        config.get('model_name'), os.path.abspath(model_filepath))

    # Setup the dataset
    dataset_directory = os.path.join(
        parent_directory, 'data', 'audio', config.get('dataset_name'))
    representation_directory = os.path.join(
        parent_directory,
        'data',
        config.get('representation_directory'),
        config.get('dataset_name'),
        config.get('model_name'))
    dataset = dataset_factory(
        config.get('dataset_name'),
        dataset_directory,
        representation_directory,
        config.get('measure_similarity_batch_size'),
        config.get('construct_representation_batch_size'),
        model)

    voogle = Voogle(model, dataset, config.get('require_text_match'))

    print('Leave-one-out loss for model {} on dataset {} is {}'.format(
        config.get('model_name'),
        config.get('dataset_name'),
        compute_loss(voogle, dataset)))
