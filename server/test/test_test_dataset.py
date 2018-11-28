import librosa
import os
import unittest
from data.TestDataset import TestDataset
from model.SiameseStyle import SiameseStyle
from model.text.ContainsText import ContainsText


class TestTestDataset(unittest.TestCase):

    def setUp(self):
        self.dataset_directory = os.path.realpath(
            'server/data/audio/test_dataset')
        self.representation_directory = os.path.realpath(
            'server/data/representations/test_dataset/siamese-style')
        self.model_filepath = os.path.realpath(
            'server/model/weights/default_model.h5')

        self.model = SiameseStyle(self.model_filepath)
        self.dataset = TestDataset(
            self.dataset_directory,
            self.representation_directory,
            self.model,
            120,
            120)

        self.query_audio, self.sr_query = librosa.load(
            os.path.join(self.dataset_directory, 'cat.wav'), sr=None)
        self.query = self.model.construct_representation(
            [self.query_audio], [self.sr_query], is_query=True)[0]

        self.text_handler = ContainsText()
        self.text_handler.set_query_text('')
        self.require_text_match = False

    def test_dataset_audio_filenames(self):
        filenames = self.dataset._get_audio_filenames()
        self.assertEqual(len(filenames), 120)
        for filename in filenames:
            self.assertEqual(filename[-4:].lower(), '.wav')

    def test_pairwise_batch_generator(self):
        handles = self.dataset._get_representation_handles()
        representations = self.dataset._load_representations(handles)
        filenames = [
            self.dataset.handle_to_filename(handle) for handle in handles]
        generator = self.dataset._pairwise_batch_generator(
            self.query, representations, filenames)
        gen_output = list(generator)

        represented_files = [False] * len(filenames)
        for query_batch, item_batch, file_tracker in gen_output:
            self.assertEqual(len(query_batch), len(item_batch))
            for filename in file_tracker.values():
                self.assertTrue(filename in filenames)
                represented_files[filenames.index(filename)] = True
            for index in file_tracker.keys():
                self.assertLess(index, len(query_batch))
        self.assertTrue(all(represented_files))

    def test_generator_default(self):
        filenames = []
        generator = self.dataset.data_generator(
            self.query, self.text_handler, self.require_text_match)
        for batch_items, batch_audio, file_tracker in generator:
            # Every representation should have a corresponding filename
            self.assertEqual(len(batch_items), len(batch_audio))

        # Every audio file should have a representation
        self.assertEqual(
            len(os.listdir(self.dataset_directory)),
            len(self.dataset.data_dict))

        # Each audio file should map to a unique representation
        for filename in filenames:
            audio_filename = os.path.join(self.dataset_directory, filename)
            representation_filename = os.path.join(
                self.representation_directory, filename) + '.npy'
            self.assertTrue(os.path.exists(audio_filename))
            self.assertTrue(os.path.exists(representation_filename))


if __name__ == '__main__':
    unittest.main()
