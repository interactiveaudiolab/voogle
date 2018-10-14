import os
import unittest
from TestDataset import TestDataset
from SiameseStyle import SiameseStyle


class TestTestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset_directory = os.path.realpath('../data/audio/test_dataset')
        self.representation_directory = os.path.realpath(
            '../data/representations/test_dataset')
        self.model_filepath = os.path.realpath(
            '../data/model/default_model.h5')

        self.dataset = TestDataset(
            self.dataset_directory, self.representation_directory)
        self.model = SiameseStyle()
        self.model.load_model(self.model_filepath)

    def test_generator_default(self):
        generator = self.dataset(self.model)

        i = 0
        for batch, filenames in generator:
            i += 1
            # Every representation should have a corresponding filename
            self.assertEqual(len(batch), len(filenames))

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

        # TODO: more tests


if __name__ == '__main__':
    unittest.main()
