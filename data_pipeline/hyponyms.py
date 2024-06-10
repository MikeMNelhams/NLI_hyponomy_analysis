from __future__ import annotations
import random
from warnings import warn
from typing import Dict

import math
import numpy as np
from nltk.corpus import wordnet as wn
import NLI_hyponomy_analysis.data_pipeline.file_operations as file_op


class Hyponyms(file_op.DictWriter):
    def __init__(self, hyponyms_file_path: str, unique_words: list, depth: int=10):
        super(Hyponyms, self).__init__(hyponyms_file_path)

        if not self.file_exists or self.file_empty:
            self.hyponyms = self.hyponyms_from_words(unique_words, depth=depth)

            self.save(self.hyponyms)
        else:
            self.hyponyms = self.load()
            self.hyponyms = {key: value for key, value in self.hyponyms.items() if key in unique_words}

    def __repr__(self):
        return str(self.hyponyms)

    @staticmethod
    def hyponyms_from_words(word_list, pos=None, depth=10) -> dict:

        def hypo(_s):
            return _s.hyponyms()
        total = 0
        hyponyms = {word: [] for word in word_list}
        count = 0
        for word in word_list:
            count += 1
            if count % 100 == 0:
                print(f"Got the hyponyms of {0} words out of {1}".format(count, len(word_list)))
            # get synsets of word
            synset_list = wn.synsets(word, pos=pos)
            if len(synset_list) > 0:
                for synset in synset_list:
                    # collect all the synsets below a given synset
                    synsets = list(synset.closure(hypo, depth=depth))
                    # include the synset itself as well
                    synsets.append(synset)
                    for s in synsets:
                        for ln in s.lemma_names():
                            hyponyms[word].append(ln.lower())
                hyponyms[word] = list(set(hyponyms[word]))
            else:
                hyponyms[word] = 'OOV'
                total += 1
        print(f"From all the words, {100*total/count}% are NOT hyponyms")
        return hyponyms


class DenseHyponymMatrices:
    def __init__(self, hyponyms: Hyponyms,
                 embedding_vectors: Dict[str, np.array],
                 normalisation=False):
        self.normalisation = normalisation

        if embedding_vectors is None:
            raise TypeError(f"hyponym_vectors cannot be None!")

        if hyponyms is not None:
            # We don't want to save/load this. Since time to load scales O(n^2). But time to generate is O(n).
            self.density_matrices = self.__density_matrices(hyponyms.hyponyms, embedding_vectors)
        else:
            self.density_matrices = embedding_vectors

        # Remove empty string (sometimes contained in big GloVe vector lists)
        try:
            del self.density_matrices[""]
        except KeyError:
            pass

        # Filter the 'OOV' pairs
        self.density_matrices = {key: value for key, value in self.density_matrices.items() if type(value) != str}
        self.d_emb = self.__get_d_emb(self.density_matrices)

    def __len__(self):
        return len(self.density_matrices)

    def __repr__(self):
        return str(self.density_matrices.keys())

    @classmethod
    def from_file(cls, file_path: str) -> DenseHyponymMatrices:
        obj = cls.__new__(cls)
        file_writer = file_op.DictWriter(file_path)

        if not file_writer.file_exists or file_writer.file_empty:
            raise FileNotFoundError

        density_matrices = file_writer.load()
        density_matrices = {key: np.array(value) for key, value in density_matrices.items()}

        try:
            del density_matrices[""]
        except KeyError:
            pass

        obj.density_matrices = density_matrices
        obj.d_emb = cls.__get_d_emb(density_matrices)
        obj.normalisation = False
        return obj

    def __density_matrices(self, hyponym_dict: dict, hyponym_vectors: dict) -> Dict[str, np.array]:
        dim = len(list(hyponym_vectors.values())[0])  # dim = length of first vector, all same
        vectors = {word: np.zeros([dim, dim]) for word in hyponym_dict.keys()}

        self._assert_valid_normalisation_type()

        for word in hyponym_dict:
            if hyponym_dict[word] == 'OOV':
                if word not in hyponym_vectors:
                    continue
                v = hyponym_vectors[word]
                vectors[word] = np.outer(v, v)
            else:
                for hyponym in hyponym_dict[word]:
                    try:
                        if hyponym not in hyponym_vectors:
                            continue
                    except Exception as e:
                        print(hyponym)
                        raise e
                    hyponym_vector = hyponym_vectors[hyponym]
                    vectors[word] += np.outer(hyponym_vector, hyponym_vector)

            v = vectors[word]
            if np.all(v == 0):
                vectors[word] = 'OOV'
            elif not self.normalisation:
                pass
            elif self.normalisation == 'trace1':
                v /= np.trace(v)
                vectors[word] = v
            elif self.normalisation == 'maxeig1':
                v /= np.max(np.linalg.eigvalsh(v))
                vectors[word] = v

        print('Done generating hyponym vectors')
        print('-' * 50)
        return vectors

    @staticmethod
    def __get_d_emb(density_matrices: dict) -> int:
        # TODO find the error type and switch to case specific error handling
        try:
            return list(density_matrices.values())[0].shape[0] ** 2
        except:
            try:
                return density_matrices["cat"].shape[0] ** 2
            except:
                print("Something went from determining d_emb")
                raise ValueError

    def lookup(self, word: str) -> np.array:
        word = self.density_matrices.get(word, None)
        return word

    def safe_lookup(self, word: str) -> np.array:
        vector = self.lookup(word)

        if np.all(vector == 0):
            return None

        return vector

    def remove_all_except(self, words: list) -> None:
        self.density_matrices = {key: value for key, value in self.density_matrices.items() if key in words}
        return None

    def flatten(self) -> None:
        print('-' * 50)
        print("Flattening vectors...")
        self.density_matrices = {key: value.flatten() for key, value in self.density_matrices.items()}
        print("Done flattening vectors.")
        print('-' * 50)
        return None

    def square(self) -> None:
        shape = self.__shape
        assert len(shape) == 1

        square_size = int(math.sqrt(shape[0]))
        assert square_size ** 2, TypeError

        self.density_matrices = {key: value.reshape((square_size, square_size))
                                 for key, value in self.density_matrices.items()}
        return None

    def generate_missing_vectors(self, words: list, glove_vectors, pad_value=0):
        padding_vector = np.array([pad_value for _ in range(self.d_emb)])

        def get_vector(word: str) -> np.array:
            value = self.lookup(word)

            if value is not None:
                return value.flatten()

            value = glove_vectors.lookup(word)

            # Lookup returns UNK/PAD if word is OOV
            if value is None:
                return padding_vector

            try:
                value = np.outer(value, value).flatten()
            except TypeError:
                print(f"Word {word}")
                raise TypeError
            return value

        for key in words:
            self.density_matrices[key] = get_vector(key)
        return None

    def to_csv(self, file_path: str):
        if file_op.is_file(file_path):
            raise FileExistsError

        assert len(self.__shape) == 1
        csv_writer = file_op.CSV_Writer(file_path, delimiter=' ')
        lines = [[key] + value.tolist() for key, value in self.density_matrices.items()]
        csv_writer.write(lines)

    @property
    def __shape(self):
        shape = list(self.density_matrices.values())[0].shape
        return shape

    def _assert_valid_normalisation_type(self) -> None:
        if self.normalisation not in ("trace1", "maxeig1", False):
            warn('Possible arguments to normalisation are "trace1", "maxeig1" or False (default). '
                 f'You entered {self.normalisation}')
            warn('Setting normalisation to False')
            self.normalisation = False
        return None


def main():
    pass


if __name__ == "__main__":
    main()
