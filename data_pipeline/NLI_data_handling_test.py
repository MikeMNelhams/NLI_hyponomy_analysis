import os.path
import random
import time
import unittest

from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_Unclean, SNLI_DataLoader_Processed
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_POS_Processed
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import InvalidBatchKeyError, BatchSizeTooLargeError


class NLI_properties(unittest.TestCase):
    test_loader_small = SNLI_DataLoader_Unclean("../data/snli_small/snli_small1_dev.jsonl")

    def test_length(self):
        self.assertEqual(len(self.test_loader_small), 101)


class UncleanLoad(unittest.TestCase):
    test_loader = SNLI_DataLoader_Unclean("../data/snli_1.0/snli_1.0_dev.jsonl")
    test_loader_small = SNLI_DataLoader_Unclean("../data/snli_small/snli_small1_dev.jsonl")

    def test_load_random(self):
        self.test_loader._batch_index = 0
        e = self.test_loader.load_random(1000)
        for thing in e.data:
            assert type(thing) == dict

    def test_load_line(self):
        self.test_loader._batch_index = 0
        random_lines = [random.randint(0, self.test_loader.file_size) for _ in range(100)]

        for random_line_number in random_lines:
            self.test_loader.load_line(random_line_number)

    def test_load_line_beyond(self):
        self.test_loader_small._batch_index = 0
        with self.assertRaises(InvalidBatchKeyError):
            self.test_loader_small.load_line(101)

    def test_load_line_final(self):
        self.test_loader_small._batch_index = 0
        print(self.test_loader_small.load_line(100))

    def test_load_line_first(self):
        self.test_loader_small._batch_index = 0
        print(self.test_loader_small.load_line(0))

    def test_load_sequential(self):
        self.test_loader._batch_index = 0

        self.test_loader.load_sequential(101)
        self.assertEqual(self.test_loader._batch_index, 101)

    def test_load_sequential_oversized_param(self):
        self.test_loader_small._batch_index = 0
        with self.assertRaises(BatchSizeTooLargeError):
            self.assertEqual(len(self.test_loader_small.load_sequential(512)), 101)

    def test_load_sequential_exact_size_param(self):
        self.test_loader_small._batch_index = 0
        batch = self.test_loader_small.load_sequential(101)
        print(batch)
        self.assertEqual(len(batch), 101)
        self.assertEqual(self.test_loader_small._batch_index, 0)

    def test_load_sequential_wraps(self):
        self.test_loader_small._batch_index = 0
        self.test_loader_small.load_sequential(50)
        self.assertEqual(self.test_loader_small._batch_index, 50)

        self.test_loader_small.load_sequential(50)
        self.assertEqual(self.test_loader_small._batch_index, 100)

        self.test_loader_small.load_sequential(50)
        self.assertEqual(self.test_loader_small._batch_index, 49)

    def test_load_sequential_wraps_multiple_times(self):
        self.test_loader_small._batch_index = 0
        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 100)

        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 99)

        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 98)

        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 97)

    def test_load_sequential_single(self):
        self.test_loader_small._batch_index = 3

        self.test_loader_small.load_sequential(1)
        self.assertEqual(self.test_loader_small._batch_index, 4)

    def test_load_sequential_wraps_from_end_single(self):
        self.test_loader_small._batch_index = 100

        self.test_loader_small.load_sequential(1)
        self.assertEqual(self.test_loader_small._batch_index, 0)

    def test_load_sequential_wraps_from_end_small(self):
        self.test_loader_small._batch_index = 100

        self.test_loader_small.load_sequential(2)
        self.assertEqual(self.test_loader_small._batch_index, 1)

    def test_load_sequential_wraps_from_end_full(self):
        self.test_loader_small._batch_index = 100

        self.test_loader_small.load_sequential(2)
        self.assertEqual(self.test_loader_small._batch_index, 1)

    def test_load_sequential_non_wraps(self):
        self.test_loader._batch_index = 0

        self.test_loader.load_sequential(50)
        self.assertEqual(self.test_loader._batch_index, 50)

        self.test_loader.load_sequential(50)
        self.assertEqual(self.test_loader._batch_index, 100)

        self.test_loader.load_sequential(50)
        self.assertEqual(self.test_loader._batch_index, 150)

    def _range_read_time(self):
        self.test_loader._batch_index = 0
        start = time.perf_counter()
        self.test_loader.load_sequential(len(self.test_loader))
        end = time.perf_counter()
        return end - start

    def test_range_read_time(self):
        num_times = 100
        times = [self._range_read_time() for _ in range(num_times)]
        print("Mean time:", sum(times)/num_times)


class UncleanFileWriting(unittest.TestCase):
    test_loader = SNLI_DataLoader_Unclean("../data/snli_1.0/snli_1.0_dev.jsonl")

    def test_max_len_value(self):
        self.assertEqual(self.test_loader.max_words_in_sentence_length, 50)

    def test_max_len_file_exists(self):
        self.assertTrue(os.path.isfile(self.test_loader.max_len_file_path))

    def test_unique_file_exists(self):
        self.assertTrue(os.path.isfile(self.test_loader.unique_words_file_path))


class ProcessedLoad_L(unittest.TestCase):
    test_loader = SNLI_DataLoader_Processed("../data/snli_1.0/snli_1.0_dev.jsonl", "lemma")
    test_loader_small = SNLI_DataLoader_Processed("../data/snli_small/snli_small1_dev.jsonl", "lemma")

    def test_load_random(self):
        self.test_loader._batch_index = 0
        e = self.test_loader.load_random(1000)
        for thing in e.data:
            assert type(thing) == dict

    def test_load_line(self):
        self.test_loader._batch_index = 0
        random_lines = [random.randint(0, self.test_loader.file_size) for _ in range(100)]

        for random_line_number in random_lines:
            self.test_loader.load_line(random_line_number)

    def test_load_line_beyond(self):
        self.test_loader_small._batch_index = 0
        with self.assertRaises(InvalidBatchKeyError):
            self.test_loader_small.load_line(101)

    def test_load_line_final(self):
        self.test_loader_small._batch_index = 0
        print(self.test_loader_small.load_line(100))

    def test_load_line_first(self):
        self.test_loader_small._batch_index = 0
        print(self.test_loader_small.load_line(0))

    def test_load_sequential(self):
        self.test_loader._batch_index = 0

        self.test_loader.load_sequential(101)
        self.assertEqual(self.test_loader._batch_index, 101)

    def test_load_sequential_oversized_param(self):
        self.test_loader_small._batch_index = 0
        with self.assertRaises(BatchSizeTooLargeError):
            self.assertEqual(len(self.test_loader_small.load_sequential(512)), 101)

    def test_load_sequential_exact_size_param(self):
        self.test_loader_small._batch_index = 0
        batch = self.test_loader_small.load_sequential(101)
        print(batch)
        self.assertEqual(len(batch), 101)
        self.assertEqual(self.test_loader_small._batch_index, 0)

    def test_load_sequential_wraps(self):
        self.test_loader_small._batch_index = 0
        self.test_loader_small.load_sequential(50)
        self.assertEqual(self.test_loader_small._batch_index, 50)

        self.test_loader_small.load_sequential(50)
        self.assertEqual(self.test_loader_small._batch_index, 100)

        self.test_loader_small.load_sequential(50)
        self.assertEqual(self.test_loader_small._batch_index, 49)

    def test_load_sequential_wraps_multiple_times(self):
        self.test_loader_small._batch_index = 0
        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 100)

        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 99)

        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 98)

        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 97)

    def test_load_sequential_single(self):
        self.test_loader_small._batch_index = 3

        self.test_loader_small.load_sequential(1)
        self.assertEqual(self.test_loader_small._batch_index, 4)

    def test_load_sequential_wraps_from_end_single(self):
        self.test_loader_small._batch_index = 100

        self.test_loader_small.load_sequential(1)
        self.assertEqual(self.test_loader_small._batch_index, 0)

    def test_load_sequential_wraps_from_end_small(self):
        self.test_loader_small._batch_index = 100

        self.test_loader_small.load_sequential(2)
        self.assertEqual(self.test_loader_small._batch_index, 1)

    def test_load_sequential_wraps_from_end_full(self):
        self.test_loader_small._batch_index = 100

        self.test_loader_small.load_sequential(2)
        self.assertEqual(self.test_loader_small._batch_index, 1)

    def test_load_sequential_non_wraps(self):
        self.test_loader._batch_index = 0

        self.test_loader.load_sequential(50)
        self.assertEqual(self.test_loader._batch_index, 50)

        self.test_loader.load_sequential(50)
        self.assertEqual(self.test_loader._batch_index, 100)

        self.test_loader.load_sequential(50)
        self.assertEqual(self.test_loader._batch_index, 150)


class ProcessedFileWriting_L(unittest.TestCase):
    test_loader = SNLI_DataLoader_Processed("../data/snli_1.0/snli_1.0_dev.jsonl", "lemma")

    def test_max_len_value(self):
        self.assertEqual(self.test_loader.max_words_in_sentence_length, 42)

    def test_max_len_file_exists(self):
        self.assertTrue(os.path.isfile(self.test_loader.max_len_file_path))

    def test_unique_file_exists(self):
        self.assertTrue(os.path.isfile(self.test_loader.unique_words_file_path))


class ProcessedLoad_POS(unittest.TestCase):
    test_loader = SNLI_DataLoader_POS_Processed("../data/snli_1.0/snli_1.0_dev.jsonl")
    test_loader_small = SNLI_DataLoader_POS_Processed("../data/snli_small/snli_small1_dev.jsonl")

    def test_load_random(self):
        self.test_loader._batch_index = 0
        e = self.test_loader.load_random(1000)
        for thing in e.data:
            assert type(thing) == dict

    def test_load_line(self):
        self.test_loader._batch_index = 0
        random_lines = [random.randint(0, self.test_loader.file_size) for _ in range(100)]

        for random_line_number in random_lines:
            self.test_loader.load_line(random_line_number)

    def test_load_line_beyond(self):
        self.test_loader_small._batch_index = 0
        with self.assertRaises(InvalidBatchKeyError):
            self.test_loader_small.load_line(101)

    def test_load_line_final(self):
        self.test_loader_small._batch_index = 0
        print(self.test_loader_small.load_line(100))

    def test_load_line_first(self):
        self.test_loader_small._batch_index = 0
        print(self.test_loader_small.load_line(0))

    def test_load_sequential(self):
        self.test_loader._batch_index = 0

        self.test_loader.load_sequential(101)
        self.assertEqual(self.test_loader._batch_index, 101)

    def test_load_sequential_oversized_param(self):
        self.test_loader_small._batch_index = 0
        with self.assertRaises(BatchSizeTooLargeError):
            self.assertEqual(len(self.test_loader_small.load_sequential(512)), 101)

    def test_load_sequential_exact_size_param(self):
        self.test_loader_small._batch_index = 0
        batch = self.test_loader_small.load_sequential(101)
        print(batch)
        self.assertEqual(len(batch), 101)
        self.assertEqual(self.test_loader_small._batch_index, 0)

    def test_load_sequential_wraps(self):
        self.test_loader_small._batch_index = 0
        self.test_loader_small.load_sequential(50)
        self.assertEqual(self.test_loader_small._batch_index, 50)

        self.test_loader_small.load_sequential(50)
        self.assertEqual(self.test_loader_small._batch_index, 100)

        self.test_loader_small.load_sequential(50)
        self.assertEqual(self.test_loader_small._batch_index, 49)

    def test_load_sequential_wraps_multiple_times(self):
        self.test_loader_small._batch_index = 0
        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 100)

        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 99)

        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 98)

        self.test_loader_small.load_sequential(100)
        self.assertEqual(self.test_loader_small._batch_index, 97)

    def test_load_sequential_single(self):
        self.test_loader_small._batch_index = 3

        self.test_loader_small.load_sequential(1)
        self.assertEqual(self.test_loader_small._batch_index, 4)

    def test_load_sequential_wraps_from_end_single(self):
        self.test_loader_small._batch_index = 100

        self.test_loader_small.load_sequential(1)
        self.assertEqual(self.test_loader_small._batch_index, 0)

    def test_load_sequential_wraps_from_end_small(self):
        self.test_loader_small._batch_index = 100

        self.test_loader_small.load_sequential(2)
        self.assertEqual(self.test_loader_small._batch_index, 1)

    def test_load_sequential_wraps_from_end_full(self):
        self.test_loader_small._batch_index = 100

        self.test_loader_small.load_sequential(2)
        self.assertEqual(self.test_loader_small._batch_index, 1)

    def test_load_sequential_non_wraps(self):
        self.test_loader._batch_index = 0

        self.test_loader.load_sequential(50)
        self.assertEqual(self.test_loader._batch_index, 50)

        self.test_loader.load_sequential(50)
        self.assertEqual(self.test_loader._batch_index, 100)

        self.test_loader.load_sequential(50)
        self.assertEqual(self.test_loader._batch_index, 150)


class ProcessedFileWriting_POS(unittest.TestCase):
    test_loader = SNLI_DataLoader_POS_Processed("../data/snli_1.0/snli_1.0_dev.jsonl")

    def test_max_len_value(self):
        self.assertEqual(self.test_loader.max_words_in_sentence_length, 49)

    def test_max_len_file_exists(self):
        self.assertTrue(os.path.isfile(self.test_loader.max_len_file_path))

    def test_unique_file_exists(self):
        self.assertTrue(os.path.isfile(self.test_loader.unique_words_file_path))


if __name__ == '__main__':
    unittest.main()
