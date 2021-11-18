import os

# noinspection PyUnresolvedReferences
from embeddings import GloveEmbedding
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import sqlite3


def glove_matrix(input_file_path: str, output_file_path: str):

    if not os.path.isfile(output_file_path):
        print('Generating all word2vec vectors.')
        glove2word2vec(glove_input_file=input_file_path, word2vec_output_file=output_file_path)
        print('Done')
        print('-' * 50)

    print('Loading keyed vectors')
    out = KeyedVectors.load_word2vec_format(output_file_path, binary=False)
    print('Done')
    print('-' * 50)
    return out


def remove_all_non_unique(word_vectors: GloveEmbedding, unique_words: list) -> None:
    """ https://docs.python.org/3/library/sqlite3.html """

    # unique_words_table = [(word, index) for index, word in enumerate(unique_words)]
    #
    # unique_words_db = sqlite3.connect(":memory:")
    #
    # # Create the table
    # unique_words_db.execute("create table uniqwords(word, indexcole)")
    #
    # # Fill the table
    # unique_words_db.executemany("insert into uniqwords(word, indexcole) values (?, ?)", unique_words_table)
    #
    # # Print table contents
    # for row in unique_words_db.execute("select word, indexcole from uniqwords"):
    #     print(row)

    c = word_vectors.db.cursor()

    c.execute("select * from embeddings")
    print(len(c.fetchall()))

    c.execute(f"delete from embeddings where word not in {tuple(unique_words)}")

    c.execute("select * from embeddings")
    print(len(c.fetchall()))

    c.close()
    return None

