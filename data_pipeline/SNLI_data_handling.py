import os.path
from typing import List
from typing import Iterable
from typing import Any

from itertools import chain
from collections import Counter
from collections import OrderedDict

import random
import numpy as np

import json

from word_operations import WordParser, count_max_sequence_length
from data_pipeline.file_operations import file_path_without_extension
from nltk.tokenize import word_tokenize
from embeddings import GloveEmbedding

import torch


class BatchSizeTooLargeError(Exception):
    """ When the specified batch size > file size"""
    def __init__(self, batch_size: int, file_size):

        message = f"The batch size \'{batch_size}\' is greater than the file_size \'{file_size}\'."
        super().__init__(message)


class NotSingleFieldError(Exception):
    """ When you try to do something to a batch which hasn't be trimmed to a single field"""


class InvalidBatchKeyError(Exception):
    pass


class PadSizeTooSmallError(Exception):
    """Called when pad size < list size"""


class Batch:
    def __init__(self, list_batch: List[Any]):
        self.data = list_batch
        self.is_single_field = False

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()


class SentenceBatch(Batch):
    """ Loaded with useful stats properties"""
    def __init__(self, list_of_sentences: List[str]):
        super().__init__(list_of_sentences)
        self.data = list_of_sentences

    @property
    def word_frequency(self):
        return self.__get_unique_words()

    @property
    def unique_words(self):
        return sorted(tuple(set(self.word_frequency.keys())))

    @property
    def number_of_unique_words(self):
        return len(self.unique_words)

    def __get_unique_words(self) -> OrderedDict:
        sentences_list = [word_tokenize(sentence) for sentence in self.data]
        sentences_list = chain(*sentences_list)
        words_list = Counter(sentences_list)
        del sentences_list
        return OrderedDict(sorted(words_list.items()))


class GoldLabelBatch(Batch):
    def __init__(self, list_of_labels: List[str]):
        super().__init__(list_of_labels)
        self.data = list_of_labels


class EntailmentModelBatch:
    """ [[sentence1: str], [sentence2: str], [label: str]]"""
    """ Entailment is either 'contradiction': -1, 'neutral': 0, 'entailment': 1"""

    class_label_encoding = {'entailment': 0,
                      'neutral': 1,
                      'contradiction': 2,
                      '-': 1}

    def __init__(self, sentence1_batch: Iterable, sentence2_batch: Iterable, labels: Iterable,
                 max_sequence_len: int,
                 word_delimiter=' '):
        self.data = np.array((sentence1_batch, sentence2_batch, labels)).T
        self.batch_size = self.data.shape[0]
        self.num_sentences = self.data.shape[1] - 1
        self.__word_delimiter = word_delimiter

        self.__max_sequence_len = max_sequence_len

        self.__labels_encoding = self.__get_labels_encoding()

    def __str__(self):
        return str(self.data)

    @property
    def word_delimiter(self):
        return self.__word_delimiter

    @property
    def labels_encoding(self):
        return self.__labels_encoding

    def __max_sentence_length(self, sentence_column: np.array) -> int:
        return max(len(line.split(self.word_delimiter)) for line in sentence_column)

    def clean_data(self, clean_actions: WordParser = None) -> None:
        # Ternary operator for SPEED and lack of intelliJ errors
        clean = np.vectorize(clean_actions) if clean_actions is not None else np.vectorize(WordParser.default_clean())

        for col_index in range(self.data.shape[1]):
            self.data[:, col_index] = clean(self.data[:, col_index])
        return None

    @staticmethod
    def pad(list_to_pad: list, max_length: int, pad_value: Any = 0) -> list:
        if len(list_to_pad) > max_length:
            raise PadSizeTooSmallError
        padded_list = list_to_pad + [pad_value for _ in range(max_length - len(list_to_pad))]
        return padded_list

    def to_tensors(self, word_vectors: GloveEmbedding, pad_value=0):
        # Make empty lists
        sentences = [None for _ in range(self.data.shape[1] - 1)]
        masks = sentences.copy()

        # Fetch all the tensor info for each batch of sentences.
        for i in range(len(sentences)):
            sentences[i], masks[i] = self.__sentence_to_tensors(sentence_num=i + 1, word_vectors=word_vectors,
                                                                pad_value=pad_value)

        sentences, masks = self.__sentence_tensor_stack(sentences, masks, pad_value=pad_value)
        sentences, masks = self.__permute_tensor(sentences), self.__permute_tensor(masks)
        return sentences, masks

    @staticmethod
    def __permute_tensor(tensor: torch.Tensor):
        return tensor.permute(0, 3, 1, 2)

    def __sentence_to_tensors(self, sentence_num: int,
                              word_vectors: GloveEmbedding, pad_value=0) -> (torch.Tensor, torch.Tensor):
        """ word_vectors must be same length for all words.
            sentence_num begins 1, 2, 3..."""
        assert 0 < sentence_num < self.data.shape[1], "Sentence number must be less than self.data.shape[1]"

        column_number = sentence_num - 1

        data_to_process = self.data[:, column_number]

        embed_vector_length = len(list(word_vectors.lookup('the')))

        padding_list = [pad_value for _ in range(embed_vector_length)]

        def get_vector(word: Any) -> list:
            if word == 0:
                return padding_list
            word_vector = word_vectors.emb(word)
            # Lookup returns None if word is OOV
            if word_vector is None:
                return padding_list
            return word_vector

        def pad_row(row: str, __pad_value=pad_value) -> List:
            return self.pad(row.split(), self.__max_sequence_len, pad_value=__pad_value)

        padded_tensor = torch.tensor([[get_vector(word)
                                       for word in pad_row(row)]
                                     for row in data_to_process], dtype=torch.float32)

        padding_mask_tensor = torch.tensor([[1 if word != 0 else 0
                                            for word in pad_row(row, 0)]
                                            for row in data_to_process])

        desired_mask_shape = (-1, -1, embed_vector_length)

        padding_mask_tensor = padding_mask_tensor.unsqueeze(-1).expand(*desired_mask_shape)

        return padded_tensor, padding_mask_tensor

    def __sentence_tensor_stack(self,
                                sentences,
                                masks,
                                pad_value=0) -> (torch.tensor, torch.tensor):
        # Sentences/Masks INPUT will be shapes:
        # 1. (256, Mp1, 300)
        # 2. (256, Mp2, 300), ...

        paddings = tuple([sentence.shape[1] for sentence in sentences])
        biggest_sentence = int(np.argmax(paddings))
        max_pad = paddings[biggest_sentence]

        for sentence_idx in range(len(sentences)):
            if sentence_idx != biggest_sentence:
                sentences[sentence_idx] = self.__pad_tensor(sentences[sentence_idx],
                                                            max_pad=max_pad, pad_value=pad_value)

        for mask_idx in range(len(masks)):
            if mask_idx != biggest_sentence:
                masks[mask_idx] = self.__pad_tensor(masks[mask_idx], max_pad=max_pad, pad_value=0)

        # Sentences/Masks now all shape (256, Mp_{max}, 300)
        # We want to stack along new dim. Output shape -> (256, Mp_{max}, 300, number_of_sentences=2)
        return torch.stack(tuple(sentences), dim=-1), torch.stack(tuple(masks), dim=-1)

    def __pad_tensor(self, tensor_input, max_pad: int, pad_value: float = 0):
        # Input shape (b, Mp1, e)
        # Output Shape (b, Mp_{max}, e)
        embed_size = tensor_input.shape[2]
        pad = tensor_input.shape[1]
        if pad_value != 0:
            pad_tensor = torch.ones((self.batch_size, max_pad - pad, embed_size))
            pad_tensor = torch.multiply(pad_tensor, pad_value)
            return torch.concat((tensor_input, pad_tensor), dim=1)
        pad_tensor = torch.zeros((self.batch_size, max_pad - pad, embed_size))
        return torch.concat((tensor_input, pad_tensor), dim=1)

    def __get_labels_encoding(self) -> torch.tensor:
        label_column_number = self.data.shape[1] - 1
        one_hot_labels = torch.tensor([self.class_label_encoding[label] for label in self.data[:, label_column_number]])
        if one_hot_labels.shape[0] == 1:
            return torch.squeeze(one_hot_labels)
        return one_hot_labels


class DictBatch(Batch):
    sentence_fields = ('sentence1', 'sentence2', 'sentence{1,2}_parse', 'sentence{1,2}_binary_parse')

    def __init__(self, list_of_dicts: List[dict], max_sequence_len):
        super().__init__(list_of_dicts)
        self.data = list_of_dicts
        self.headers = self.data[0].keys()
        self.max_sequence_len = max_sequence_len

    def to_sentence_batch(self, field_name: str) -> SentenceBatch:
        if field_name not in self.sentence_fields:
            raise InvalidBatchKeyError
        return SentenceBatch([line[field_name] for line in self.data])

    def to_labels_batch(self, label_key_name='gold_label') -> GoldLabelBatch:
        return GoldLabelBatch([line[label_key_name] for line in self.data])

    def to_model_data(self, model_fields=('sentence1', 'sentence2', 'gold_label')) -> EntailmentModelBatch:
        sentence1_list = self.to_sentence_batch(model_fields[0]).data
        sentence2_list = self.to_sentence_batch(model_fields[1]).data
        labels = self.to_labels_batch(model_fields[2]).data

        return EntailmentModelBatch(sentence1_list, sentence2_list, labels, self.max_sequence_len)

    def count_max_words_for_sentence_field(self, field_name: str) -> int:
        if field_name not in self.sentence_fields:
            raise TypeError
        return count_max_sequence_length([line[field_name] for line in self.data])


class SNLI_DataLoader:
    def __init__(self, file_path: str, max_sequence_length=None):
        self.__file_path = file_path
        self.__file_dir_path = os.path.dirname(file_path)
        self.__max_len_save_path = file_path_without_extension(self.__file_path) + '_max_len.txt'

        self.file_size = self.__get_number_lines()

        self.__batch_index = 0

        # TODO allow sentences =/= 2
        self.num_sentences = 2

        self.max_words_in_sentence_length = 0
        if max_sequence_length is None:
            if os.path.isfile(self.__max_len_save_path):
                self.max_words_in_sentence_length = self.__load_max_sentence_len()
            else:
                self.max_words_in_sentence_length = self.__find_max_sentence_len()
                self.__save_max_sentence_len()
        else:
            self.max_words_in_sentence_length = max_sequence_length

    def __len__(self):
        return self.file_size

    def __find_max_sentence_len(self, batch_size: int=1000) -> int:
        max_len = 0
        file_load_size = min(batch_size, len(self))
        number_of_iterations = (len(self) // file_load_size) + 1
        # TODO allow sentences =/= 2
        for i in range(number_of_iterations):
            print(f'Iter: {i} of {number_of_iterations}')
            print('MAX LEN:', max_len)
            print('-'*20)
            batch = self._load_batch_sequential(file_load_size)
            sentence1_max_len = batch.count_max_words_for_sentence_field('sentence1')
            sentence2_max_len = batch.count_max_words_for_sentence_field('sentence2')
            max_len = max((sentence1_max_len, sentence2_max_len, max_len))

        return max_len

    def __save_max_sentence_len(self) -> None:
        with open(self.__max_len_save_path, 'w') as outfile:
            outfile.write(str(self.max_words_in_sentence_length))

        return None

    def __load_max_sentence_len(self) -> int:
        with open(self.__max_len_save_path, 'r') as infile:
            content = infile.read()
        return int(content)

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

        return DictBatch([content], max_sequence_len=self.max_words_in_sentence_length)

    def _load_batch_sequential(self, batch_size: int, from_start: bool =False) -> DictBatch:
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
            raise BatchSizeTooLargeError(batch_size, len(self))

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
            content = [x for i, x in enumerate(file) if i in batch_range]

        content2 = [json.loads(json_string) for json_string in content]
        del content

        self.__batch_index = batch_end_index

        if overlap:
            remaining_batch_size = batch_size - (batch_end_index - batch_start_index)
            content3 = self._load_batch_sequential(remaining_batch_size, from_start=True)
            return DictBatch(content2 + content3.data, max_sequence_len=self.max_words_in_sentence_length)

        return DictBatch(content2, max_sequence_len=self.max_words_in_sentence_length)

    def _load_batch_random(self, batch_size: int) -> DictBatch:
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

        return DictBatch(buffer, max_sequence_len=self.max_words_in_sentence_length)

    def load_clean_batch_sequential(self, batch_size: int, from_start: bool=False) -> EntailmentModelBatch:
        batch_data = self._load_batch_sequential(batch_size, from_start=from_start).to_model_data()
        batch_data.clean_data()
        return batch_data

    def load_clean_batch_random(self, batch_size: int) -> EntailmentModelBatch:
        batch_data = self._load_batch_random(batch_size).to_model_data()
        batch_data.clean_data()
        return batch_data


if __name__ == "__main__":
    pass
