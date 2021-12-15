import os
import random
from warnings import warn
from typing import Dict

import numpy as np
from nltk.corpus import wordnet as wn

from file_operations import DictWriter, list_files_in_directory


class KS:
    def __init__(self, ks_file_path: str):
        self.__file_path = ks_file_path
        self.__phrase_pairs = self.__load_ks2016()

        self.scores = list(self.__phrase_pairs.keys())
        self.words = self.get_vocab()

    @property
    def file_path(self):
        return self.__file_path

    def __load_ks2016(self) -> dict:
        print(f"Getting phrase pairs and entailments values for file {self.file_path}...")
        scores = {}
        with open(self.file_path, 'r') as f:
            for line in f:
                entries = [word.lower() for phrase in line.split(',') for word in phrase.split()]
                scores[tuple(entries[:-1])] = entries[-1]
        print(f"Done getting phrase pairs for file {self.file_path}")
        print('-' * 50)
        return scores

    def get_vocab(self) -> list:
        print("Creating vocab list from phrase pairs...")
        word_list = [word for phrases in self.scores for word in phrases]
        vocab = list(set(word_list))
        print("Done")
        print('-' * 50)
        return vocab


class Hyponyms(DictWriter):
    def __init__(self, hyponyms_file_path: str):
        super(Hyponyms, self).__init__(hyponyms_file_path)

        if not self.file_exists or self.file_empty:
            self.hyponyms = self.generate_hyponyms()

            self.save(self.hyponyms)
        else:
            self.hyponyms = self.load()

    def generate_hyponyms(self, phrase_pair_directory: str = 'data/KS2016/'):
        phrases_file_names = list_files_in_directory(phrase_pair_directory, extension_type='.txt')

        words = []
        for phrase_pair_file_name in phrases_file_names:
            phrase_pair_file_path = phrase_pair_directory + phrase_pair_file_name
            phrase_pairs = KS(phrase_pair_file_path)
            words += phrase_pairs.words

        return self.__hyponyms_from_words(words)

    @staticmethod
    def __hyponyms_from_words(word_list, pos=None, depth=10):

        def hypo(_s):
            return _s.hyponyms()

        hyponyms = {word: [] for word in word_list}
        count = 0
        for word in word_list:
            count += 1
            if count % 100 == 0:
                print("Got the hyponyms of {0} words out of {1}".format(count, len(word_list)))
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
        return hyponyms


class GloveVectors(DictWriter):
    def __init__(self, save_path: str, glove_vector_file_path: str, hyponyms: dict):
        super(GloveVectors, self).__init__(save_path)

        self.hyponyms = hyponyms

        if not self.file_exists or self.file_empty:
            self.vectors = self.__vectors(glove_vector_file_path)
            data_to_save = {key: value.tolist() for key, value in self.vectors.items()}
            self.save(data_to_save)
        else:
            self.vectors = self.load()
            self.vectors = {key: np.array(value) for key, value in self.vectors.items()}

    def __vectors(self, glove_vector_file_path: str, normalisation=False, weights=False, header=False,
                  vector_encoding: str='utf8') -> dict:
        assert os.path.isfile(glove_vector_file_path), FileNotFoundError(glove_vector_file_path)
        print("Generating vectors for each hyponym from vectors file")
        if weights:
            vocabulary = set((hyp[0] for word in self.hyponyms
                              for hyp in self.hyponyms[word]
                              if self.hyponyms[word] != 'OOV'))
        else:
            vocabulary = set((hyp for word in self.hyponyms
                              for hyp in self.hyponyms[word]
                              if self.hyponyms[word] != 'OOV'))
        hypo_vectors = {}
        with open(glove_vector_file_path, 'r', encoding=vector_encoding) as vf:
            if header:
                vf.readline()
            for line in vf:
                entry = line.split()
                if entry[0] in vocabulary:
                    vec = np.array([float(n) for n in entry[1:]])
                    if normalisation:
                        vec = vec / np.linalg.norm(vec)
                    hypo_vectors[entry[0]] = vec
        print("Done calculating hyponyms for each vector.")
        print('-' * 50)
        return hypo_vectors


class DenseHyponymMatrices(DictWriter):
    def __init__(self, density_matrices_file_path: str,
                 hyponyms: Hyponyms = None, hyponym_vectors: dict = None,
                 normalisation=False):
        super(DenseHyponymMatrices, self).__init__(density_matrices_file_path)
        self.__density_matrices_file_path = density_matrices_file_path
        self.normalisation = normalisation

        self.density_matrices = None

        if not self.file_exists or self.file_empty:
            if hyponyms is None or hyponym_vectors is None:
                raise TypeError(f"hyponyms and hyponym_vectors cannot be None, "
                                f"since {self.file_path} does not exist!")
            self.density_matrices = self.__density_matrices(hyponyms.hyponyms,
                                                                     hyponym_vectors)

            data_to_save = {key: value.tolist() for key, value in self.density_matrices.items()}
            self.save(data_to_save)

        else:
            self.density_matrices = self.load()
            self.density_matrices = {key: np.array(value) for key, value in self.density_matrices.items()}

    def __len__(self):
        return len(self.density_matrices)

    @property
    def density_matrices_file_path(self):
        return self.__density_matrices_file_path

    @property
    def file_exists(self):
        return os.path.isfile(self.__density_matrices_file_path)

    @property
    def file_empty(self):
        return self.file_exists and os.stat(self.density_matrices_file_path).st_size == 0

    def __density_matrices(self, hyp_dict: dict, hypo_vectors: dict) -> Dict[str, np.array]:
        dim = len(random.choice(list(hypo_vectors.values())))  # dim= length of arbitrary vector, all same
        vocab = list(hyp_dict.keys())
        vectors = {word: np.zeros([dim, dim]) for word in vocab}

        if self.normalisation not in ("trace1", "maxeig1", False):
            warn('Possible arguments to normalisation are "trace1", "maxeig1" or False (default). '
                 f'You entered {self.normalisation}')
            warn('Setting normalisation to False')
            self.normalisation = False

        for word in hyp_dict:
            if hyp_dict[word] == 'OOV':
                continue
            for hyp in hyp_dict[word]:
                if hyp not in hypo_vectors:
                    continue
                v = hypo_vectors[hyp]  # make sure I alter the Hearst code
                vv = np.outer(v, v)
                vectors[word] += vv
            v = vectors[word]
            if np.all(v == 0):
                vectors[word] = 'OOV'
                continue
            if self.normalisation == 'trace1':
                assert np.trace(v) != 0, "Trace is 0, should be OOV"
                v = v / np.trace(v)
                vectors[word] = v
            elif self.normalisation == 'maxeig1':
                maxeig = np.max(np.linalg.eigvalsh(v))
                assert maxeig != 0, "Max eigenvalue is 0, should be OOV"
                v = v / maxeig
                vectors[word] = v
            elif not self.normalisation:
                pass
        print('Done generating hyponym vectors')
        print('-' * 50)
        return vectors


def main():
    hyponyms_all = Hyponyms('../data/hyponyms/all_hyponyms.json')
    vectors = GloveVectors('../data/hyponyms/glove_vectors.json',
                           '../data/embedding_data/glove/glove.42B.300d.txt', hyponyms_all.hyponyms)
    density_matrices = DenseHyponymMatrices(density_matrices_file_path="../data/hyponyms/dm-50d-glove-wn.json",
                                            hyponyms=hyponyms_all, hyponym_vectors=vectors.vectors)

    print("We built {0} density matrices".format(len(density_matrices)))

    # print('-' * 50)
    #
    # print('~'*80, "\nDensity matrices:", density_matrices.density_matrices, '\n', '~'*80)


if __name__ == "__main__":
    main()
