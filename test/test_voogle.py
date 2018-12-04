import librosa
import os
import unittest
from model.SiameseStyle import SiameseStyle
from data.TestDataset import TestDataset
from voogle import Voogle


class TestVoogle(unittest.TestCase):
    '''
    Test cases for the Voogle class
    '''

    def setUp(self):
        model = SiameseStyle(
            os.path.realpath('model/weights/siamese_style.h5'))

        dataset_directory = os.path.realpath(
            'data/audio/test_dataset')
        representation_directory = os.path.realpath(
            'data/representations/test_dataset/siamese-style')
        dataset = TestDataset(
            dataset_directory, representation_directory, model)
        self.voogle = Voogle(model, dataset, False)

        self.query, self.sr_query = librosa.load(
            os.path.join(dataset_directory, 'cat.wav'), sr=None)

    def test_search(self):
        '''
        Test a basic query
        '''
        match_list, text_query, similarity_scores = self.voogle.search(
            self.query, self.sr_query)
        self.assertEqual(len(match_list), 15)
        self.assertEqual(len(text_query), 15)
        self.assertEqual(len(similarity_scores), 15)
        for i in range(len(similarity_scores) - 1):
            self.assertGreater(similarity_scores[i], similarity_scores[i + 1])

if __name__ == '__main__':
    unittest.main()
