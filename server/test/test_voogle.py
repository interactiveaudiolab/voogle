import librosa
import os
import unittest
from model.SiameseStyle import SiameseStyle
from data.TestDataset import TestDataset
from Voogle import Voogle


class TestVoogle(unittest.TestCase):
    '''
    Test cases for the Voogle class
    '''

    def setUp(self):
        model = SiameseStyle(
            os.path.realpath('server/model/weights/default_model.h5'))

        dataset_directory = os.path.realpath(
            'server/data/audio/test_dataset')
        representation_directory = os.path.realpath(
            'server/data/representations/test_dataset/siamese-style')
        dataset = TestDataset(
            dataset_directory, representation_directory, model)
        self.voogle = Voogle(model, dataset, matches=10)

        self.query, self.sr_query = librosa.load(
            os.path.join(dataset_directory, 'cat.wav'), sr=None)

    def test_search(self):
        '''
        Test a basic query
        '''
        match_list, text_query = self.voogle.search(
            self.query, self.sr_query)
        self.assertEqual(len(match_list), 10)
        self.assertEqual(len(text_query), 10)

if __name__ == '__main__':
    unittest.main()
