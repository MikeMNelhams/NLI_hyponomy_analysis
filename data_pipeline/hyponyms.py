import json
import os
import pickle
import random
from warnings import warn

import numpy as np
from nltk.corpus import wordnet as wn

import file_operations


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


class Hyponyms:
    def __init__(self, hyponyms_file_path: str):
        self.__hyponyms_file_path = hyponyms_file_path

        self.hyponyms = None
        if not self.file_exists or self.file_empty:
            self.hyponyms = self.generate_hyponyms()
            self.__save_hyponyms(self.hyponyms_file_path)
        else:
            self.hyponyms = self.__load_hyponyms(self.hyponyms_file_path)

    @property
    def hyponyms_file_path(self):
        return self.__hyponyms_file_path

    @property
    def file_exists(self):
        return os.path.isfile(self.hyponyms_file_path)

    @property
    def file_empty(self):
        return self.file_exists and os.stat(self.hyponyms_file_path).st_size == 0

    def __load_hyponyms(self, file_path: str) -> dict:
        if file_operations.is_file(file_path, '.p'):
            return self.__load_hyponyms_from_pickle()
        if file_operations.is_file(file_path, '.json'):
            return self.__load_hyponyms_from_JSON()

        raise file_operations.InvalidPathError

    def __save_hyponyms(self, file_path: str) -> None:
        if file_operations.is_file(file_path, '.p'):
            self.__save_hyponyms_to_pickle()
        if file_operations.is_file(file_path, '.json'):
            self.__save_hyponyms_to_JSON()

        raise file_operations.InvalidPathError

    def __load_hyponyms_from_pickle(self) -> dict:
        print("Loading hyponyms dictionary..")
        with open(self.hyponyms_file_path, 'rb') as hyp_file:
            loaded_hyponyms: dict = pickle.load(hyp_file)
        print("Finished loading hyponyms dictionary")
        print('-' * 50)
        return loaded_hyponyms

    def __load_hyponyms_from_JSON(self) -> dict:
        print("Loading dictionary of hyponyms from JSON...")
        with open(self.hyponyms_file_path, "r") as infile:
            hyponyms: dict = json.load(infile)
        print("Done loading dictionary of hyponyms from JSON")
        print('-' * 50)
        return hyponyms

    def __save_hyponyms_to_pickle(self) -> None:
        print("Pickling out dictionary of hyponyms...")
        with open(self.hyponyms_file_path, "wb") as outfile:
            pickle.dump(self.hyponyms, outfile)
        print("Done pickling dictionary of hyponyms.")
        print('-' * 50)
        return None

    def __save_hyponyms_to_JSON(self) -> None:
        print("Saving dictionary of hyponyms to JSON...")
        with open(self.hyponyms_file_path, "w") as outfile:
            json.dump(self.hyponyms, outfile)
        print("Done saving dictionary of hyponyms to JSON")
        print('-' * 50)

    def generate_hyponyms(self, phrase_pair_directory: str = 'data/KS2016/'):
        phrases_file_names = file_operations.list_files_in_directory(phrase_pair_directory, extension_type='.txt')

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

    def vectors(self, glove_vector_file_path: str, normalisation=False, weights=False, header=False,
                vector_encoding: str='utf8') -> dict:
        assert os.path.isfile(glove_vector_file_path)
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


class DenseHyponymMatrices:
    def __init__(self, hyponyms: Hyponyms, hyponym_vectors: dict, density_matrices_file_path: str, normalisation=False):
        self.__density_matrices_file_path = density_matrices_file_path
        self.normalisation = normalisation

        self.density_matrices = None
        if not self.file_exists or self.file_empty:
            self.density_matrices = self.__density_matrices(hyponyms.hyponyms,
                                                                     hyponym_vectors)
            self.__save_density_matrices_to_pickle()
        else:
            self.density_matrices = self.__load_density_matrices_from_pickle()

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

    def __density_matrices(self, hyp_dict: dict, hypo_vectors: dict):
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

    def __save_density_matrices_to_pickle(self):
        print("Pickling density matrices")
        with open(self.density_matrices_file_path, "wb") as dm_file:
            pickle.dump(self.density_matrices, dm_file)
        print("Done.")
        print('-' * 50)

    def __load_density_matrices_from_pickle(self) -> dict:
        print("Loading density matrices...")
        with open(self.density_matrices_file_path, 'rb') as dm_file:
            loaded_density_matrices: dict = pickle.load(dm_file)
        print("Finished loading density matrices")
        print('-' * 50)
        return loaded_density_matrices


def main():
    hyponyms_all = Hyponyms('../data/hyponyms/all_hyponyms.json')
    vectors = hyponyms_all.vectors('data/embedding_data/glove/glove.42B.300d.txt')
    print(vectors['a'])
    density_matrices = DenseHyponymMatrices(hyponyms_all, vectors,
                                            density_matrices_file_path="../data/hyponyms/dm-50d-glove-wn.p")

    print("We built {0} density matrices".format(len(density_matrices)))

    print('-' * 50)

    print(density_matrices.density_matrices['member'].shape)

    print('-' * 100)
    print()
    print('-' * 100)


if __name__ == "__main__":
    main()
