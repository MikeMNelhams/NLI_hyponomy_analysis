from typing import List
from typing import Any

from itertools import chain
from collections import Counter
from collections import OrderedDict

import random

import json

from nltk.tokenize import word_tokenize


class BatchSizeTooLargeError(Exception):
    """ When the specified batch size > file size"""
    def __init__(self, batch_size: int, file_size):

        message = f"The batch size \'{batch_size}\' is greater than the file_size \'{file_size}\'."
        super().__init__(message)


class NotSingleFieldError(Exception):
    """ When you try to do something to a batch which hasn't be trimmed to a single field"""


class Batch(list):
    def __init__(self, list_batch: List[Any]):
        super().__init__(list_batch)
        self.data = list_batch
        self.is_single_field = False

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()


class SentenceBatch(Batch):
    def __init__(self, list_of_sentences: List[str]):
        super().__init__(list_of_sentences)
        self.data = list_of_sentences

        # Useful info that can be gathered at init
        self.word_frequency = self.__get_unique_words()
        self.unique_words = sorted(tuple(set(self.word_frequency.keys())))
        self.number_of_unique_words = len(self.unique_words)

    def __get_unique_words(self) -> OrderedDict:
        sentences_list = [word_tokenize(sentence) for sentence in self.data]
        sentences_list = chain(*sentences_list)
        words_list = Counter(sentences_list)
        del sentences_list
        return OrderedDict(sorted(words_list.items()))


class DictBatch(Batch):
    def __init__(self, list_of_dicts: List[dict]):
        super().__init__(list_of_dicts)
        self.data = list_of_dicts
        self.headers = self.data[0].keys()

    def to_sentence_batch(self, field_name: str) -> SentenceBatch:
        return SentenceBatch([line[field_name] for line in self.data])


class SNLI_DataLoader:
    def __init__(self, file_path: str):
        self.__file_path = file_path
        self.file_size = self.__get_number_lines()

        self.__batch_index = 0

    def __get_number_lines(self) -> int:
        """ Run at init"""
        with open(self.__file_path, "r") as file:
            number_of_lines = sum([1 for i, x in enumerate(file) if x[-1] == '\n'])
        return number_of_lines

    def load_line(self, line_number: int) -> DictBatch:
        """ Only use this if you want a specific line, not a batch.

        Very efficient, better than linecache or loading entire file.

        :param line_number: int
        :return: dict
        """
        with open(self.__file_path, "r") as file:
            content = [x for i, x in enumerate(file) if i == line_number]

        content = content[0]
        content = json.loads(content)

        return DictBatch([content])

    def load_batch_sequential(self, batch_size: int, from_start: bool =False) -> DictBatch:
        """ Correct way to load data

        Very efficient

        :param from_start: bool, whether to begin from 0 or not
        :param batch_size: int
        :return: List[dict]
        """

        # For looping back to the beginning
        if from_start:
            self.__batch_index = 0

        if batch_size > self.file_size:
            raise BatchSizeTooLargeError

        if self.__batch_index >= self.file_size:
            self.__batch_index = 0

        batch_start_index = self.__batch_index
        batch_end_index = batch_start_index + batch_size

        overlap = False

        if batch_end_index > self.file_size:
            overlap = True
            batch_end_index = self.file_size - 1

        with open(self.__file_path, "r") as file:
            batch_range = range(batch_start_index, batch_end_index)
            # print('batch range:', batch_range)
            content = [x for i, x in enumerate(file) if i in batch_range]

        content2 = [json.loads(json_string) for json_string in content]
        del content

        self.__batch_index = batch_end_index

        if overlap:
            remaining_batch_size = batch_size - (batch_end_index - batch_start_index)
            content3 = self.load_batch_sequential(remaining_batch_size, from_start=True)
            return DictBatch(content2 + content3)

        return DictBatch(content2)

    def load_batch_random(self, batch_size: int) -> DictBatch:
        """ O(File_size) load random lines"""
        # Uses reservoir sampling.
        # There is actually a FASTER way to do this using more complicated sampling:
        #   https://dl.acm.org/doi/pdf/10.1145/355900.355907
        # You could try to switch the code below into a single enumeration rather than using appends for ~2x speed-up,
        #   However, I can't get my head around how to do that....

        buffer = []

        with open(self.__file_path, 'r') as f:
            for line_num, line in enumerate(f):
                n = line_num + 1.0
                r = random.random()
                if n <= batch_size:
                    buffer.append(json.loads(line))
                elif r < batch_size / n:
                    loc = random.randint(0, batch_size - 1)
                    buffer[loc] = json.loads(line)

        return DictBatch(buffer)


if __name__ == "__main__":
    test_data = SNLI_DataLoader("data/snli_1.0/snli_1.0_test.jsonl")
