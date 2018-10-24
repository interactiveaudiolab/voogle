import librosa
import os
import unittest
from data.TestDataset import TestDataset
from model.SiameseStyle import SiameseStyle


class TestTestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset_directory = os.path.realpath(
            'server/data/audio/test_dataset')
        self.representation_directory = os.path.realpath(
            'server/data/representations/test_dataset')
        self.model_filepath = os.path.realpath(
            'server/model/weights/default_model.h5')

        self.model = SiameseStyle()
        self.model.load_model(self.model_filepath)
        self.dataset = TestDataset(
            self.dataset_directory, self.representation_directory, self.model)

        self.query_audio, self.sr_query = librosa.load(
            os.path.join(dataset_directory, 'cat.wav'), sr=None)
        self.query = self.model.construct_representation(
            [query], [sampling_rate], is_query=True)[0]

    def test_dataset_audio_filenames(self):
        filenames = self.dataset._get_audio_filenames():
        assertEqual(len(filenames), 120)
        for filename in filenames:
            assertEqual(filename[-4:].lower(), '.wav')

    def test_pairwise_batch_generator(self):
        # TODO

    def test_generator_default(self):
        i = 0
        filenames = []
        generator = self.dataset.data_generator(self.query)
        for batch_items, batch_audio, file_tracker in generator:
            i += 1
            # Every representation should have a corresponding filename
            self.assertEqual(len(batch_items), len(batch_audio))
            filenames += batch_filenames

        # No batch size specified, so only one loop
        self.assertEqual(i, 1)

        # Every audio file should have a representation
        self.assertEqual(
            len(os.listdir(self.dataset_directory)),
            len(os.listdir(self.representation_directory)))

        # Each audio file should map to a unique representation
        for filename in filenames:
            audio_filename = os.path.join(self.dataset_directory, filename)
            representation_filename = os.path.join(
                self.representation_directory, filename) + '.npy'
            self.assertTrue(os.path.exists(audio_filename))
            self.assertTrue(os.path.exists(representation_filename))


if __name__ == '__main__':
    unittest.main()
