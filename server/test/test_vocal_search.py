import librosa
import os
import unittest
from SiameseStyle import SiameseStyle
from TestDataset import TestDataset
from VocalSearch import VocalSearch


class TestVocalSearch(unittest.TestCase):
    '''
    Test cases for the VocalSearch class
    '''

    def setUp(self):
        model = SiameseStyle()
        model.load_model(os.path.realpath('model/default_model.h5'))

        dataset_directory = os.path.realpath('data/audio/test_dataset')
        representation_directory = os.path.realpath(
            '../data/representations/test_dataset')
        dataset = TestDataset(dataset_directory, representation_directory)
        dataset = dataset.data_generator(model)
        self.vocal_search = VocalSearch(model, dataset)

        self.query, self.sr_query = librosa.load(
            os.path.join(dataset_directory, 'cat.wav'), sr=None)

    def test_search(self):
        '''
        Test a basic query
        '''
        filenames = self.vocal_search.search(self.query, self.sr_query)
        self.assertEqual(len(filenames), 120)
        self.assertEqual(filenames[0], 'cat.wav')


if __name__ == '__main__':
    unittest.main()
