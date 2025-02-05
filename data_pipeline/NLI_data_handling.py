from __future__ import annotations

import json
import os.path
import random
from collections import Counter, OrderedDict
from itertools import chain, islice
from typing import Any, Iterable, List

from abc import ABC, abstractmethod

import math
import numpy as np
import torch
from embeddings import GloveEmbedding
from nltk.tokenize import word_tokenize

import csv
import NLI_hyponomy_analysis.data_pipeline.file_operations as file_op
import NLI_hyponomy_analysis.data_pipeline.word_operations as word_op
from NLI_hyponomy_analysis.data_pipeline.word_operations import WordParser, ProcessingSynonyms, standardise_label
from NLI_hyponomy_analysis.comp_analysis_library.policies import Policy
from NLI_hyponomy_analysis.comp_analysis_library.parse_tree import ParseTree


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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


class InvalidBatchMode(Exception):
    def __init__(self, mode):
        valid_modes = NLI_DataLoader_abc.batch_modes
        self.message = f"The given mode: {mode} is not a valid mode. Try using one of: {valid_modes}"
        super().__init__(self.message)


class Batch(list):
    def __init__(self, list_batch: List[Any]):
        super(Batch, self).__init__(list_batch)
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
    def word_frequency(self) -> OrderedDict:
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

    def clean(self, clean_actions: WordParser = WordParser.default_clean()) -> None:
        self.data = [clean_actions(row) for row in self.data]
        return None


class GoldLabelBatch(Batch):
    def __init__(self, list_of_labels: List[str]):
        super().__init__(list_of_labels)
        self.data = list_of_labels

    @property
    def label_count(self) -> OrderedDict:
        words_list: dict = Counter(self.data)
        return OrderedDict(sorted(words_list.items()))

    @property
    def label_frequency(self) -> OrderedDict:
        label_count = self.label_count
        frequencies = np.array(list(label_count.values()))
        freq_dict = OrderedDict([(label, freq) for label, freq in zip(self.standard_labels, frequencies)])
        if "unknown" not in label_count:
            freq_dict.update({"unknown": 0})
            freq_dict.move_to_end("unknown", last=False)
        return freq_dict

    @property
    def standard_labels(self) -> list:
        parser = WordParser((str.lower, str.strip))
        parser.append(standardise_label)
        labels = list(self.label_count.keys())
        labels = [parser(label) for label in labels]
        return labels


class EntailmentModelBatch:
    """ [[sentence1: str], [sentence2: str], [label: str]]"""

    def __init__(self, sentence1_batch: Iterable, sentence2_batch: Iterable, labels: Iterable,
                 max_sequence_len: int, word_delimiter=' '):
        self.device = device
        self.class_label_encoding = {'entailment': 0,
                                     'neutral': 1,
                                     'contradiction': 2,
                                     'contradictio': 2,
                                     '-': 3}

        self.data = np.array((sentence1_batch, sentence2_batch, labels)).T
        self.batch_size = self.data.shape[0]
        self.num_sentences = self.data.shape[1] - 1
        self.__word_delimiter = word_delimiter

        self.__max_sequence_len = max_sequence_len

        self.__labels_encoding = self.__get_labels_encoding()

        # For __iter__
        self.index = 0

    def __str__(self):
        return str(self.data)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.data.shape[0]:
            result = self.data[self.index, :]
            self.index += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.data[item, :]

    @property
    def word_delimiter(self):
        return self.__word_delimiter

    @property
    def labels_encoding(self):
        return self.__labels_encoding

    def __max_sentence_length(self, sentence_column: np.array) -> int:
        return max(len(line.split(self.word_delimiter)) for line in sentence_column)

    def process(self, processing_type: str = "clean", processing_actions=None):
        if processing_type in ProcessingSynonyms.synonyms_for_l:
            return self.lemmatise_data(processing_actions)
        if processing_type in ProcessingSynonyms.synonyms_for_cl:
            return self.lemmatise_data(self.clean_data(processing_actions))
        if processing_type in ProcessingSynonyms.synonyms_for_l_pos:
            return self.lemmatise_data_pos(processing_actions)
        if processing_type in ProcessingSynonyms.synonyms_for_l_pos_tag:
            return self.lemmatise_data_pos_tag(processing_actions)

        return self.clean_data(processing_actions)

    def clean_data(self, clean_actions: WordParser = None) -> None:
        clean = np.vectorize(WordParser.default_clean())
        if clean_actions is not None:
            clean = np.vectorize(clean_actions)

        for col_index in range(self.data.shape[1] - 1):
            self.data[:, col_index] = clean(self.data[:, col_index])
        return None

    def lemmatise_data_pos(self, lemmatise_actions: WordParser = None) -> None:
        process = np.vectorize(WordParser.default_lemmatisation_pos())
        if lemmatise_actions is not None:
            process = np.vectorize(lemmatise_actions)

        for col_index in range(self.data.shape[1] - 1):
            self.data[:, col_index] = process(self.data[:, col_index])
        return None

    def lemmatise_data_pos_tag(self, lemmatise_actions: WordParser = None) -> None:
        """ Returns the POS form"""
        process = np.vectorize(WordParser.default_lemmatisation_pos_tag())
        if lemmatise_actions is not None:
            process = np.vectorize(lemmatise_actions)

        for col_index in range(self.data.shape[1] - 1):
            self.data[:, col_index] = process(self.data[:, col_index])
        return None

    def lemmatise_data(self, lemmatise_actions: WordParser = None) -> None:
        process = np.vectorize(WordParser.default_lemmatisation())
        if lemmatise_actions is not None:
            process = np.vectorize(lemmatise_actions)

        for col_index in range(self.data.shape[1] - 1):
            self.data[:, col_index] = process(self.data[:, col_index])
        return None

    def append_to_file(self, file_path: str) -> None:
        with open(file_path, "a", newline='', encoding="utf-8") as out_file:
            writer = csv.writer(out_file)

            writer.writerows(self.data)
        return None

    @staticmethod
    def pad(list_to_pad: list, max_length: int, pad_value: Any = 0) -> list:
        if len(list_to_pad) > max_length:
            raise PadSizeTooSmallError
        padded_list = list_to_pad + [pad_value for _ in range(max_length - len(list_to_pad))]
        return padded_list

    def to_tensors(self, word_vectors: GloveEmbedding, pad_value=0, max_length=None):
        # Make empty lists
        sentences = [None for _ in range(self.data.shape[1] - 1)]

        masks = sentences.copy()

        # Fetch all the tensor info for each batch of sentences.
        for i in range(len(sentences)):
            sentences[i], masks[i] = self.__sentence_to_tensors(sentence_num=i + 1, word_vectors=word_vectors,
                                                                pad_value=pad_value)

        sentences, masks = self.__sentence_tensor_stack(sentences, masks, pad_value=pad_value, max_length=max_length)
        return sentences, masks

    def __sentence_to_tensors(self, sentence_num: int, word_vectors: GloveEmbedding,
                              pad_value=0) -> (torch.Tensor, torch.Tensor):
        """ word_vectors must be same length for all words.
            sentence_num begins 1, 2, 3..."""
        assert 0 < sentence_num < self.data.shape[1], "Sentence number must be less than self.data.shape[1]"

        column_number = sentence_num - 1

        data_to_process = self.data[:, column_number]

        embed_vector_length = word_vectors.d_emb

        padding_list = [pad_value for _ in range(embed_vector_length)]

        unknown_word_vector = word_vectors.lookup('<unk>')
        if unknown_word_vector is None:
            unknown_word_vector = padding_list

        padded_tensor = torch.tensor([[self.__get_vector(word, word_vectors, padding_list, unknown_word_vector)
                                       for word in self.__pad_row(row, pad_value)]
                                      for row in data_to_process], dtype=torch.float32, device=self.device)

        padding_mask_tensor = torch.tensor([[1 if word != 0 else 0
                                             for word in self.__pad_row(row, pad_value)]
                                            for row in data_to_process], device=self.device)

        desired_mask_shape = (-1, -1, embed_vector_length)

        padding_mask_tensor = padding_mask_tensor.unsqueeze(-1).expand(*desired_mask_shape)

        return padded_tensor, padding_mask_tensor

    def __sentence_tensor_stack(self, sentences, masks, pad_value=0, max_length=None) -> (torch.tensor, torch.tensor):
        # Sentences/Masks INPUT will be shapes:
        # 1. (256, Mp1, 300)
        # 2. (256, Mp2, 300), ...
        # We output shape (256, Mp_{max}, 300)
        if max_length is not None:
            max_pad = max_length

            for sentence_idx in range(len(sentences)):
                sentences[sentence_idx] = self.__pad_tensor(sentences[sentence_idx],
                                                            max_pad=max_pad, pad_value=pad_value)

            for mask_idx in range(len(masks)):
                masks[mask_idx] = self.__pad_tensor(masks[mask_idx], max_pad=max_pad, pad_value=0)

        else:
            paddings = tuple(sentence.shape[1] for sentence in sentences)
            longest_sentence_index = int(np.argmax(paddings))
            max_pad = paddings[longest_sentence_index]

            for sentence_idx in range(len(sentences)):
                if sentence_idx != longest_sentence_index:
                    sentences[sentence_idx] = self.__pad_tensor(sentences[sentence_idx],
                                                                max_pad=max_pad, pad_value=pad_value)

            for mask_idx in range(len(masks)):
                if mask_idx != longest_sentence_index:
                    masks[mask_idx] = self.__pad_tensor(masks[mask_idx], max_pad=max_pad, pad_value=0)

        # Sentences/Masks now all shape (256, Mp_{max}, 300)
        # We want to stack along new dim. Output shape -> (256, number_of_sentences=2, Mp_{max}, 300)
        return torch.stack(tuple(sentences), dim=1), torch.stack(tuple(masks), dim=1)

    def __pad_tensor(self, tensor_input, max_pad: int, pad_value: float = 0):
        # Input shape (b, Mp1, e)
        # Output Shape (b, Mp_{max}, e)
        embed_size = tensor_input.shape[2]
        pad = tensor_input.shape[1]
        if pad_value != 0:
            pad_tensor = torch.ones((self.batch_size, max_pad - pad, embed_size), device=self.device)
            pad_tensor = torch.multiply(pad_tensor, pad_value)
            return torch.concat((tensor_input, pad_tensor), dim=1)
        pad_tensor = torch.zeros((self.batch_size, max_pad - pad, embed_size), device=self.device)
        return torch.concat((tensor_input, pad_tensor), dim=1)

    def __get_labels_encoding(self) -> torch.tensor:
        label_column_number = self.data.shape[1] - 1

        label_encodings = [self.class_label_encoding[label]
                           if label != ''
                           else self.class_label_encoding['-']
                           for label in self.data[:, label_column_number]]

        one_hot_labels = torch.tensor(label_encodings, device=self.device)
        if one_hot_labels.shape[0] == 1:
            return torch.squeeze(one_hot_labels)
        return one_hot_labels

    @staticmethod
    def __get_vector(word: Any, word_vectors, padding_list, unknown_word_vector) -> list:
        if word == 0:
            return padding_list
        word_vector = word_vectors.lookup(word)
        # Lookup returns UNK/PAD if word is OOV
        if word_vector is None:
            return list(unknown_word_vector)
        return list(word_vector)

    def __pad_row(self, row: str, pad_value=0) -> List:
        padded = self.pad(row.split(), self.__max_sequence_len, pad_value=pad_value)
        return padded


class PolicyEvaluatedBatch:
    """ [[sentence1: POS TAGGED str], [sentence2: POS TAGGED str], [label: POS TAGGED str]]"""
    def __init__(self, sentence1_list: List[str], sentence2_list: List[str], labels: Iterable,
                 word_vectors, policy: Policy):
        self.device = device
        self.class_label_encoding = {'entailment': 0,
                                     'neutral': 1,
                                     'contradiction': 2,
                                     'contradictio': 2,
                                     '-': 3}

        self.word_vectors = word_vectors
        self.policy = policy

        self.data = np.array((self._apply_policy(sentence1_list), self._apply_policy(sentence2_list), labels)).T
        self.batch_size = len(sentence1_list)
        self.num_sentences = self.data.shape[1] - 1

        self.__labels_encoding = self.__get_labels_encoding()

        # For __iter__
        self.index = 0

    def __str__(self):
        return str(self.data)

    def __next__(self):
        if self.index < self.data.shape[0]:
            result = self.data[self.index, :]
            self.index += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.data[item, :]

    def __get_labels_encoding(self) -> torch.tensor:
        label_column_number = self.data.shape[1] - 1

        label_encodings = [self.class_label_encoding[label]
                           if label != ''
                           else self.class_label_encoding['-']
                           for label in self.data[:, label_column_number]]

        one_hot_labels = torch.tensor(label_encodings, device=self.device)
        if one_hot_labels.shape[0] == 1:
            return torch.squeeze(one_hot_labels)
        return one_hot_labels

    def _apply_policy(self, batch: Iterable) -> List[np.array]:
        parse_trees = [ParseTree(sentence, self.word_vectors, self.policy) for sentence in batch]
        for parse_tree in parse_trees:
            parse_tree.evaluate()
        parse_trees = [parse_tree.data[0] for parse_tree in parse_trees]
        return parse_trees


class DictBatch(Batch):
    sentence_fields = ("sentence1", "sentence2", "sentence{1,2}_parse", "sentence{1,2}_binary_parse", "sentence1_parse",
                       "sentence2_parse", "sentence1_binary_parse", "sentence2_binary_parse")

    def __init__(self, list_of_dicts: List[dict], max_sequence_len):
        super().__init__(list_of_dicts)
        try:
            self.headers = self[0].keys()
        except IndexError:
            print('DATA:', self)
            raise ZeroDivisionError
        self.max_sequence_len = max_sequence_len

    def __len__(self):
        return len(self)

    def to_sentence_batch(self, field_name: str) -> SentenceBatch:
        self.assert_valid_sentence(field_name)
        return SentenceBatch([self._get_sentence(line, field_name) for line in self])

    def to_batch(self, field_name: str):
        self.assert_valid_sentence(field_name)
        return Batch([self._get_sentence(line, field_name) for line in self])

    def to_labels_batch(self, label_key_name='gold_label') -> GoldLabelBatch:
        for line in self:
            try:
                line["gold_label"]
            except Exception as e:
                print("bad batch: ", line)
                raise e

        labels = GoldLabelBatch([line[label_key_name] for line in self])
        return labels

    def to_model_data(self, model_fields=('sentence1', 'sentence2', 'gold_label')) -> EntailmentModelBatch:
        sentence1_list = self.to_batch(model_fields[0]).data
        sentence2_list = self.to_batch(model_fields[1]).data
        labels = self.to_labels_batch(model_fields[2]).data

        return EntailmentModelBatch(sentence1_list, sentence2_list, labels, self.max_sequence_len)

    def apply_policy(self, word_vectors, policy: Policy, model_fields=('sentence1', 'sentence2', 'gold_label')):
        """ Only use for POS tagged sentences"""
        sentence1_list = self.to_batch(model_fields[0]).data
        sentence2_list = self.to_batch(model_fields[1]).data
        labels = self.to_labels_batch(model_fields[2]).data

        return PolicyEvaluatedBatch(sentence1_list, sentence2_list, labels, word_vectors, policy)

    def count_max_words_for_sentence_field(self, field_name: str) -> int:
        if field_name not in self.sentence_fields:
            raise TypeError
        try:
            sequence = [line[field_name] if line else '' for line in self]
            max_length = word_op.count_max_sequence_length(sequence)
            return max_length
        except KeyError as e:
            print(self)
            raise e

    @staticmethod
    def _get_sentence(line, field_name: str):
        if not line:
            return ''
        return line[field_name]

    def assert_valid_sentence(self, field_name: str) -> None:
        if field_name not in self.sentence_fields:
            raise InvalidBatchKeyError
        return None


class UniqueWords:
    def __init__(self, data_loader: NLI_DataLoader_abc):
        self.data_loader = data_loader
        self.unique_words_file_path = data_loader.unique_words_file_path

    @property
    def file_exists(self):
        return os.path.isfile(self.unique_words_file_path)

    def get_unique_words(self) -> list:
        if self.file_exists:
            return self.load_unique_words()

        unique_words = self.__unique_words()
        self.save_unique_words(unique_words)
        return unique_words

    def __unique_words(self) -> list:
        print('-' * 50)
        print("Generating unique words list...")
        print("Loading data...")
        train_data = self.data_loader.load_all()
        print("Data loaded")
        unique_words1 = train_data.to_sentence_batch("sentence1")
        unique_words2 = train_data.to_sentence_batch("sentence2")
        print("Performing set operations...")
        unique_words1 = unique_words1.unique_words
        unique_words2 = unique_words2.unique_words
        total_unique_words = set(unique_words1).union(set(unique_words2)) - word_op.pos_tags
        all_unique_words = sorted(list(total_unique_words))
        print("Finished set operations")
        print("Unique word list made.")
        print('-'*50)
        return all_unique_words

    def save_unique_words(self, unique_words) -> None:
        with open(self.unique_words_file_path, "w", encoding="utf-8") as out_file:
            writer = csv.writer(out_file)
            writer.writerow(unique_words)
        return None

    def load_unique_words(self) -> list:
        with open(self.unique_words_file_path, "r", encoding="utf-8") as in_file:
            reader = csv.reader(in_file)
            unique_words = list(reader)
        return unique_words[0]


class NLI_DataLoader_abc(ABC):
    batch_modes = ('sequential', "random")

    def __init__(self, file_path: str, *args, dict_batch_constructor=DictBatch, **kwargs):
        self.file_path = file_path
        self.file_dir_path = file_op.file_path_without_extension(file_path) + '/'
        self.unique_words_file_path = self.file_dir_path + "unique.csv"
        self.max_len_file_path = self.file_dir_path + 'max_len.txt'

        self.file_load_path = file_path

        self.max_words_in_sentence_length = 0
        self._max_sentence_len_writer = file_op.TextWriterSingleLine(self.max_len_file_path)

        self._make_dir()

        self.file_size = None
        # TODO make this derive from the data given
        self.num_sentences = 2
        self._batch_index = 0

        self.dict_batch_constructor = dict_batch_constructor

    def __len__(self):
        return self.file_size

    def _make_dir(self) -> None:
        if file_op.is_dir(self.file_dir_path):
            return None

        file_op.make_dir(self.file_dir_path)
        return None

    def is_valid_batch_mode(self, mode: str) -> bool:
        return mode in self.batch_modes

    def load_all(self) -> DictBatch:
        data = self.load_sequential(len(self))
        return data

    @property
    def file_exists(self):
        return os.path.isfile(self.file_path)

    def term_count(self, column_name: str = "sentence1") -> OrderedDict:
        train_data = self.load_all()
        train_data = train_data.to_sentence_batch(column_name)

        term_count = train_data.word_frequency
        return term_count

    def label_count(self) -> OrderedDict:
        batch = self.load_sequential(len(self)).to_labels_batch()
        label_count = batch.label_count
        return label_count

    def _max_sequence_len(self, batch_size: int = 1_000) -> int:
        max_len = 0
        file_load_size = min(batch_size, len(self))
        number_of_iterations = (len(self) // file_load_size) + 1
        # TODO allow sentences =/= 2
        for i in range(number_of_iterations):
            print(f'Iter: {i} of {number_of_iterations}')
            print('MAX LEN:', max_len)
            print('-' * 20)
            batch = self.load_sequential(file_load_size)
            sentence1_max_len = batch.count_max_words_for_sentence_field('sentence1')
            sentence2_max_len = batch.count_max_words_for_sentence_field('sentence2')
            max_len = max((sentence1_max_len, sentence2_max_len, max_len))

        return max_len

    def _get_max_sequence_length(self) -> int:
        if os.path.isfile(self.max_len_file_path):
            max_length = int(self._max_sentence_len_writer.load())
            return max_length
        else:
            max_length = self._max_sequence_len()
            self._max_sentence_len_writer.save(max_length)
        return max_length

    def load_batch(self, batch_size: int, mode: str, **kwargs) -> DictBatch:
        if mode == "sequential":
            return self.load_sequential(batch_size, **kwargs)
        if mode == "random":
            return self.load_random(batch_size)
        raise InvalidBatchMode

    def load_line(self, line_number: int) -> DictBatch:
        """ Only use this if you want a specific line, not a batch.

        Very efficient, better than linecache or loading entire file.

        :param line_number: int
        :return: dict
        """
        content = self.__read_line(line_number)

        return self.dict_batch_constructor(content, max_sequence_len=self.max_words_in_sentence_length)

    def load_random(self, batch_size: int = 256, default=-1) -> DictBatch:
        """ O(File_size) load_as_dataframe random lines
            Highly optimised using reservoir sampling L-algorithm (Optimal)
            https://en.wikipedia.org/wiki/Reservoir_sampling
        """

        with open(self.file_load_path, "r") as iterable:
            reservoir = list(map(self._parse_file_line, islice(iterable, batch_size)))

            w = math.exp(math.log(random.random()) / batch_size)

            while True:
                s = math.floor(math.log(random.random()) / math.log(1 - w))
                next_elem = next(islice(iterable, s + 1, None), default)
                if next_elem == default:
                    return self.dict_batch_constructor(reservoir, max_sequence_len=self.max_words_in_sentence_length)

                reservoir[random.randrange(0, batch_size)] = self._parse_file_line(next_elem)
                w *= math.exp(math.log(random.random()) / batch_size)

    def load_sequential(self, batch_size: int = 256, from_start: bool = False) -> DictBatch:
        if from_start:
            self._batch_index = 0

        if batch_size > self.file_size:
            raise BatchSizeTooLargeError(batch_size, len(self))

        start_index = self._batch_index
        end_index = self._batch_index + batch_size

        overlap = False
        if end_index > self.file_size:
            overlap = True
            end_index = self.file_size

        content1 = self.__read_range(start_index, end_index)

        if overlap:
            end_index_wrapped = batch_size - (self.file_size - start_index)
            content2 = self.__read_range(0, end_index_wrapped)
            self._batch_index = end_index_wrapped
            return self.dict_batch_constructor(content1 + content2, max_sequence_len=self.max_words_in_sentence_length)

        if end_index == self.file_size:
            self._batch_index = 0
        else:
            self._batch_index = end_index

        return self.dict_batch_constructor(content1, max_sequence_len=self.max_words_in_sentence_length)

    @abstractmethod
    def _parse_file_line(self, lines: list):
        raise NotImplementedError

    def __assert_valid_line_number(self, line_number: int) -> None:
        if line_number >= self.file_size or line_number < 0:
            raise InvalidBatchKeyError
        return None

    def __read_line(self, line_number: int) -> list:
        self.__assert_valid_line_number(line_number)

        with open(self.file_load_path, "r") as file:
            content = [self._parse_file_line(x) for i, x in enumerate(file) if i == line_number]

        return content

    def __read_range(self, start_index: int, end_index: int) -> list:
        if start_index == end_index:
            return self.__read_line(start_index)

        assert end_index <= self.file_size, InvalidBatchKeyError

        # Makes use of early stopping AND known list memory allocation.
        with open(self.file_load_path, "r") as file:
            batch_range = range(start_index, end_index)
            content = [{} for _ in batch_range]
            for i, x in enumerate(file):
                if i >= end_index:
                    break
                if i >= start_index:
                    content[i - start_index] = self._parse_file_line(x)

        return content


class SNLI_DataLoader_Unclean(NLI_DataLoader_abc):
    def __init__(self, file_path: str, max_sequence_length=None):
        super(SNLI_DataLoader_Unclean, self).__init__(file_path, max_sequence_length=max_sequence_length)

        # Run once at runtime, rather than multiple times at call.
        self.file_size = file_op.count_file_lines(file_path)

        if max_sequence_length is None:
            self.max_words_in_sentence_length = self._get_max_sequence_length()
        else:
            self.max_words_in_sentence_length = max_sequence_length

        self.unique_words = UniqueWords(self).get_unique_words()

    def _parse_file_line(self, line: str):
        return json.loads(line)


class SNLI_DataLoader_Processed(NLI_DataLoader_abc):
    def __init__(self, file_path: str, processing_mode: str, max_sequence_length=None,
                 processing_batch_size: int = 256):
        super(SNLI_DataLoader_Processed, self).__init__(file_path, max_sequence_length=max_sequence_length)

        self.__processing_type = ProcessingSynonyms.map_processing_mode(processing_mode)

        self.file_dir_path += self.processing_type + '/'
        self.processed_file_path = self.file_dir_path + 'processed.csv'
        self.unique_words_file_path = self.file_dir_path + 'unique.csv'
        self.max_len_file_path = self.file_dir_path + 'max_len.txt'

        self.file_load_path = self.processed_file_path

        self._make_dir()

        self._max_sentence_len_writer = file_op.TextWriterSingleLine(self.max_len_file_path)

        # Run once at runtime, rather than multiple times at call.
        self.file_size = file_op.count_file_lines(self.processed_file_path)

        if not self.processed_file_exists:
            file_op.make_empty_file_safe(self.processed_file_path)

            self.__process_and_save_data(self.processing_type, processing_batch_size=processing_batch_size)

        if max_sequence_length is None:
            self.max_words_in_sentence_length = self._get_max_sequence_length()
        else:
            self.max_words_in_sentence_length = max_sequence_length

        # Run once at runtime, rather than multiple times at call.
        self.unique_words = UniqueWords(self).get_unique_words()

    @property
    def processing_type(self) -> str:
        return self.__processing_type

    @property
    def processed_file_exists(self):
        return os.path.isfile(self.processed_file_path)

    def _parse_file_line(self, line: str) -> dict:
        """ Enumerating lines reads as a single string with a \n on the end. That needs fixing"""
        out_row = line.split(',')
        if out_row[-1][-1] == '\n':
            out_row[-1] = out_row[-1][:-1]  # Remove the \n
        out_dict = {"sentence1": out_row[0], "sentence2": out_row[1], "gold_label": out_row[-1]}
        return out_dict

    def __temporary_batch_size_schedule(self, batch_size) -> [int]:
        """ [256, 256, 3] for example of size 515"""
        number_of_fixed_batch_size_steps = len(self) // batch_size

        remainder_batch_size = len(self) - number_of_fixed_batch_size_steps * batch_size
        batch_sizes = [batch_size for _ in range(number_of_fixed_batch_size_steps + 1)]
        batch_sizes[-1] = remainder_batch_size

        if number_of_fixed_batch_size_steps * batch_size == len(self):
            batch_sizes.pop()

        return batch_sizes

    def __process_and_save_data(self, processing_type, processing_batch_size: int = 256) -> None:
        """ Called once at first instantiation. Slow, slow overhead. Reduces load_as_dataframe time massively."""

        unclean_data_loader = SNLI_DataLoader_Unclean(self.file_path)

        batch_sizes = self.__temporary_batch_size_schedule(processing_batch_size)
        num_iters = len(batch_sizes)

        # Stops if it has looped across entire dataset.
        for i, batch_size in enumerate(batch_sizes):
            print(f"Processing batch ~{i+1} of {num_iters}...")
            data_batch = unclean_data_loader.load_sequential(batch_size)
            data_batch = data_batch.to_model_data()

            data_batch.process(processing_type)

            data_batch.append_to_file(self.processed_file_path)

        file_op.trim_end_of_file_blank_line(self.processed_file_path)

        return None


class SNLI_DataLoader_POS_tagged(NLI_DataLoader_abc):
    """ Returns the sentences in POS form"""
    def __init__(self, file_path: str, processing_batch_size: int = 256):
        super(SNLI_DataLoader_POS_tagged, self).__init__(file_path)

        self.file_dir_path += 'lemmatised_pos_tagged' + '/'
        self.processed_file_path = self.file_dir_path + 'processed.csv'
        self.unique_words_file_path = self.file_dir_path + 'unique.csv'
        self.max_len_file_path = self.file_dir_path + 'max_len.txt'

        self.file_load_path = self.processed_file_path

        self._make_dir()

        self._max_sentence_len_writer = file_op.TextWriterSingleLine(self.max_len_file_path)

        self._batch_index = 0

        if not self.processed_file_exists:
            self.file_size = file_op.count_file_lines(self.file_path)
            file_op.make_empty_file_safe(self.processed_file_path)
            self.__process_and_save_data(processing_batch_size=processing_batch_size)

        self.file_size = file_op.count_file_lines(self.processed_file_path)
        self.max_words_in_sentence_length = 1

        # Run once at runtime, rather than multiple times at call.
        self.unique_words = UniqueWords(self).get_unique_words()

    def __temporary_batch_size_schedule(self, batch_size) -> [int]:
        """ [256, 256, 3] for example of size 515"""
        number_of_fixed_batch_size_steps = len(self) // batch_size

        remainder_batch_size = len(self) - number_of_fixed_batch_size_steps * batch_size
        batch_sizes = [batch_size for _ in range(number_of_fixed_batch_size_steps + 1)]
        batch_sizes[-1] = remainder_batch_size

        if number_of_fixed_batch_size_steps * batch_size == len(self):
            batch_sizes.pop()

        return batch_sizes

    def __process_and_save_data(self, processing_batch_size: int = 1_000):
        """ Called once at first instantiation. Slow, slow overhead. Reduces load_as_dataframe time massively."""

        unclean_data_loader = SNLI_DataLoader_Unclean(self.file_path)

        batch_sizes = self.__temporary_batch_size_schedule(processing_batch_size)
        num_iters = len(batch_sizes)

        # Stops if it has looped across entire dataset.
        for i, batch_size in enumerate(batch_sizes):
            print(f"Processing batch ~{i+1} of {num_iters}...")
            data_batch = unclean_data_loader.load_sequential(batch_size)
            data_batch = data_batch.to_model_data(['sentence1_parse', 'sentence2_parse', 'gold_label'])

            data_batch.process('lemmatised_pos_tag')

            data_batch.append_to_file(self.processed_file_path)
        file_op.trim_end_of_file_blank_line(self.processed_file_path)

        return None

    @property
    def processed_file_exists(self):
        return os.path.isfile(self.processed_file_path)

    def _parse_file_line(self, line: str) -> dict:
        """ Enumerating lines reads as a single string with a \n on the end. That needs fixing"""
        out_row = line.split(',')
        if out_row[-1][-1] == '\n':
            out_row[-1] = out_row[-1][:-1]  # Remove the \n
        out_dict = {"sentence1": out_row[0], "sentence2": out_row[1], "gold_label": out_row[-1]}
        return out_dict

    def _max_sequence_len(self, batch_size: int = 1_000) -> int:
        return 1


class SNLI_DataLoader_POS_Processed(NLI_DataLoader_abc):
    def __init__(self, file_path: str, max_sequence_length=None, processing_batch_size: int = 256):
        super(SNLI_DataLoader_POS_Processed, self).__init__(file_path)

        self.file_dir_path += 'lemmatised_pos' + '/'
        self.processed_file_path = self.file_dir_path + 'processed.csv'
        self.unique_words_file_path = self.file_dir_path + 'unique.csv'
        self.max_len_file_path = self.file_dir_path + 'max_len.txt'

        self.file_load_path = self.processed_file_path

        self._make_dir()

        self._max_sentence_len_writer = file_op.TextWriterSingleLine(self.max_len_file_path)

        self._batch_index = 0

        if not self.processed_file_exists:
            self.file_size = file_op.count_file_lines(self.file_path)
            file_op.make_empty_file_safe(self.processed_file_path)
            self.__process_and_save_data(processing_batch_size=processing_batch_size)

        self.file_size = file_op.count_file_lines(self.processed_file_path)

        if max_sequence_length is None:
            self.max_words_in_sentence_length = self._get_max_sequence_length()
        else:
            self.max_words_in_sentence_length = max_sequence_length

        # Run once at runtime, rather than multiple times at call.
        self.unique_words = UniqueWords(self).get_unique_words()

        # Keep all the TEXT data in RAM ~ 0.5 GB
        self.data = self.load_all()

    def load_batch(self, batch_size: int, mode: str, **kwargs) -> DictBatch:
        if mode == "sequential":
            return self.load_sequential_faster(batch_size, **kwargs)
        if mode == "random":
            return self.load_random(batch_size)
        raise InvalidBatchMode

    def load_sequential_faster(self, batch_size: int, from_start: bool=False):
        if from_start:
            self._batch_index = 0

        if batch_size > self.file_size:
            raise BatchSizeTooLargeError(batch_size, len(self))

        start_index = self._batch_index
        end_index = self._batch_index + batch_size

        overlap = False
        if end_index > self.file_size:
            overlap = True
            end_index = self.file_size

        content1 = self.__read_range_faster(start_index, end_index)

        if overlap:
            end_index_wrapped = batch_size - (self.file_size - start_index)
            content2 = self.__read_range_faster(0, end_index_wrapped)
            self._batch_index = end_index_wrapped
            return self.dict_batch_constructor(content1 + content2, max_sequence_len=self.max_words_in_sentence_length)

        if end_index == self.file_size:
            self._batch_index = 0
        else:
            self._batch_index = end_index

        return self.dict_batch_constructor(content1, max_sequence_len=self.max_words_in_sentence_length)

    def __temporary_batch_size_schedule(self, batch_size) -> [int]:
        """ [256, 256, 3] for example of size 515"""
        number_of_fixed_batch_size_steps = len(self) // batch_size

        remainder_batch_size = len(self) - number_of_fixed_batch_size_steps * batch_size
        batch_sizes = [batch_size for _ in range(number_of_fixed_batch_size_steps + 1)]
        batch_sizes[-1] = remainder_batch_size

        if number_of_fixed_batch_size_steps * batch_size == len(self):
            batch_sizes.pop()

        return batch_sizes

    def __process_and_save_data(self, processing_batch_size: int = 1_000):
        """ Called once at first instantiation. Slow, slow overhead. Reduces load_as_dataframe time massively."""

        unclean_data_loader = SNLI_DataLoader_Unclean(self.file_path)

        batch_sizes = self.__temporary_batch_size_schedule(processing_batch_size)
        num_iters = len(batch_sizes)

        # Stops if it has looped across entire dataset.
        for i, batch_size in enumerate(batch_sizes):
            print(f"Processing batch ~{i+1} of {num_iters}...")
            data_batch = unclean_data_loader.load_sequential(batch_size)
            data_batch = data_batch.to_model_data(['sentence1_parse', 'sentence2_parse', 'gold_label'])

            data_batch.process('lemmatised_pos')

            data_batch.append_to_file(self.processed_file_path)

        file_op.trim_end_of_file_blank_line(self.processed_file_path)

        return None

    @property
    def processed_file_exists(self):
        return os.path.isfile(self.processed_file_path)

    def _parse_file_line(self, line: str) -> dict:
        """ Enumerating lines reads as a single string with a \n on the end. That needs fixing"""
        out_row = line.split(',')
        if out_row[-1][-1] == '\n':
            out_row[-1] = out_row[-1][:-1]  # Remove the \n
        out_dict = {"sentence1": out_row[0], "sentence2": out_row[1], "gold_label": out_row[-1]}
        return out_dict

    def __read_range_faster(self, start_index: int, end_index: int) -> list:
        if start_index == end_index:
            return self.__read_line_faster(start_index)
        return self.data[start_index:end_index]

    def __read_line_faster(self, line_number: int) -> list:
        return self.data[line_number]


if __name__ == "__main__":
    pass
