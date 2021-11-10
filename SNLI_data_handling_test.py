import random
import unittest

from NLI_hyponomy_analysis.SNLI_data_handling import SNLI_DataLoader


class LoadBatches(unittest.TestCase):
    test_data = SNLI_DataLoader("data/snli_1.0/snli_1.0_test.jsonl")

    def test_load_batch_sequential(self):
        for i in range(100):
            assert len(self.test_data._load_batch_sequential(1010)) == 1010

    def test_load_batch_random(self):
        e = self.test_data._load_batch_random(1000)
        for thing in e.data:
            assert type(thing) == dict

    def test_load_line(self):

        randoms = [random.randint(0, self.test_data.file_size) for _ in range(100)]

        for random_line_number in randoms:
            self.test_data.load_line(random_line_number)


if __name__ == '__main__':
    unittest.main()
