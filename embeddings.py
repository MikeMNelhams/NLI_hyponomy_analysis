from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

import os


def glove_matrix(input_file_path: str, output_file_path: str):
    if not os.path.isfile(output_file_path):
        glove2word2vec(glove_input_file=input_file_path, word2vec_output_file=output_file_path)

    return KeyedVectors.load_word2vec_format(output_file_path, binary=False)
