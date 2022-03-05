import os

# noinspection PyUnresolvedReferences
from embeddings import GloveEmbedding
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import numpy as np


# Here is a table of different lookup methods I have tested
# -----------------------------------------------------------------------------------------------------#
#    SQLite (Hybrid Disk)               | 120ms - 1000ms QUERY | 002 seconds LOAD | 0   RAM | 6gb disk #
#    Load from file to RAM              | 100ms - 0200ms QUERY | 180 seconds LOAD | 5gb RAM | 5gb disk #
#    SQLite to RAM - current solution   | 020ms - 0040ms QUERY | 012 seconds LOAD | 6gb RAM | 6gb disk #
# -----------------------------------------------------------------------------------------------------#


class Embedding2(GloveEmbedding):
    def __init__(self, name='common_crawl_840', d_emb=300, show_progress=True, default='none'):
        super(Embedding2, self).__init__(name=name, d_emb=d_emb, show_progress=show_progress, default=default)

    @property
    def head(self):
        c = self.db.cursor()
        q = c.execute('select word from embeddings limit 5').fetchall()
        return q if q else None

    @property
    def words(self):
        c = self.db.cursor()
        q = c.execute("select word from embeddings").fetchall()
        return [word for row in q for word in row] if q else None


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


def remove_all_except(word_vectors: GloveEmbedding, unique_words: list) -> None:
    """ https://docs.python.org/3/library/sqlite3.html """

    c = word_vectors.db.cursor()

    c.execute(f"delete from embeddings where word not in {tuple(unique_words)}")

    # c.execute("select * from embeddings")
    # print(len(c.fetchall()))

    c.close()
    return None
