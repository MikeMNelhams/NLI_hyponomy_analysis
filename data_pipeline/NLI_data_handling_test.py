import os.path
import random
import unittest

from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_Unclean


class UncleanLoadBatches(unittest.TestCase):
    test_loader = SNLI_DataLoader_Unclean("../data/snli_1.0/snli_1.0_dev.jsonl")

    def test_load_batch_sequential(self):
        for i in range(100):
            assert len(self.test_loader.load_sequential(1010)) == 1010

    def test_load_batch_random(self):
        e = self.test_loader.load_random(1000)
        for thing in e.data:
            assert type(thing) == dict

    def test_load_line(self):
        randoms = [random.randint(0, self.test_loader.file_size) for _ in range(100)]

        for random_line_number in randoms:
            self.test_loader.load_line(random_line_number)


class UncleanFileWriting(unittest.TestCase):
    test_loader = SNLI_DataLoader_Unclean("../data/snli_1.0/snli_1.0_dev.jsonl")

    def test_max_len_value(self):
        self.assertEqual(self.test_loader.max_words_in_sentence_length, 50)

    def test_max_len_file_exists(self):
        self.assertTrue(os.path.isfile(self.test_loader.max_len_file_path))

    def test_unique_file_exists(self):
        self.assertTrue(os.path.isfile(self.test_loader.unique_words_file_path))


if __name__ == '__main__':
    unittest.main()
