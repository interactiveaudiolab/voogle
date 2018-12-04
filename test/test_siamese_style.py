import librosa
import math
import numpy as np
import os
import unittest
from data.TestDataset import TestDataset
from model.SiameseStyle import SiameseStyle


class TestSiameseStyle(unittest.TestCase):

    def setUp(self):
        self.dataset_directory = os.path.realpath(
            'data/audio/test_dataset')
        self.representation_directory = os.path.realpath(
            'data/representations/test_dataset/siamese-style')
        self.model_filepath = os.path.realpath(
            'model/weights/siamese_style.h5')
        self.model = SiameseStyle(self.model_filepath)

        # Make sure the test dataset has been downloaded
        self.dataset = TestDataset(
            self.dataset_directory,
            self.representation_directory,
            self.model,
            120,
            120)

        self.cat, self.sr_cat = librosa.load(
            os.path.join(self.dataset_directory, 'cat.wav'), sr=None)
        self.dog, self.sr_dog = librosa.load(
            os.path.join(self.dataset_directory, 'dog_barking.wav'), sr=None)

    def test_construct_representation(self):
        dataset = self.model.construct_representation(
            [self.cat, self.dog], [self.sr_cat, self.sr_dog], is_query=False)
        # The number of representations should match the number of audio clips
        self.assertEqual(len(dataset), 2)
        for datum in dataset:
            # The representations should be normalized
            self.assertTrue(math.isclose(datum.mean(), 0.0, abs_tol=1e-06))
            self.assertTrue(math.isclose(datum.std(), 1.0, abs_tol=1e-06))

        query = self.model.construct_representation(
            [self.cat], [self.sr_cat], is_query=True)
        # The number of representations should match the number of audio clips
        self.assertEqual(len(query), 1)
        # The representations should be normalized
        self.assertTrue(math.isclose(query[0].mean(), 0.0, abs_tol=1e-06))
        self.assertTrue(math.isclose(query[0].std(), 1.0, abs_tol=1e-06))

    def test_measure_similarity(self):
        dataset = self.model.construct_representation(
            [self.cat, self.dog], [self.sr_cat, self.sr_dog], is_query=False)
        query = self.model.construct_representation(
            [self.cat], [self.sr_cat], is_query=True)

        dataset = np.concatenate(dataset)
        query = np.repeat(
            np.array(query), len(dataset), axis=0)

        similarity = self.model.measure_similarity(query, dataset)

        self.assertEqual(len(similarity), len(dataset))


if __name__ == '__main__':
    unittest.main()
