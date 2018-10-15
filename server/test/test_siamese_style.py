import librosa
import math
import os
import unittest
from SiameseStyle import SiameseStyle


class TestSiameseStyle(unittest.TestCase):

    def setUp(self):
        self.model_filepath = os.path.realpath('model/default_model.h5')
        self.model = SiameseStyle()
        self.model.load_model(self.model_filepath)

        dataset_directory = os.path.realpath('data/audio/test_dataset')

        self.cat, self.sr_cat = librosa.load(
            os.path.join(dataset_directory, 'cat.wav'), sr=None)
        self.dog, self.sr_dog = librosa.load(
            os.path.join(dataset_directory, 'dog_barking.wav'), sr=None)

    def test_get_name(self):
        self.assertEqual('siamese-style', self.model.get_name())

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

    def test_predict(self):
        dataset = self.model.construct_representation(
            [self.cat, self.dog], [self.sr_cat, self.sr_dog], is_query=False)
        query = self.model.construct_representation(
            [self.cat], [self.sr_cat], is_query=True)

        predictions = self.model.predict(query, dataset)

        # Make sure the model can handle trivial cases
        self.assertTrue(predictions[0] > predictions[1])

if __name__ == '__main__':
    unittest.main()
