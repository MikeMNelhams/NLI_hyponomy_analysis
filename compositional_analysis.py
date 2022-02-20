from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_Unclean
import NLI_hyponomy_analysis.data_pipeline.file_operations as file_op

import NLI_hyponomy_analysis.data_pipeline.matrix_operations.hyponymy_library as hl
from NLI_hyponomy_analysis.data_pipeline.word_operations import find_all_pos_tags
from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed
from dotenv import load_dotenv
import numpy as np


def ke_multiply(data_path: str):
    load_dotenv()  # Path to the glove data directory -> HOME="..."

    data_loader = SNLI_DataLoader_Unclean(data_path)

    word_vectors_0 = embed.GloveEmbedding('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()
    embed.remove_all_non_unique(word_vectors_0,  data_loader.unique_words)

    word_vectors = DenseHyponymMatrices("data/hyponyms/dm-25d-glove-wn_train_lemma_pos.json")
    word_vectors.remove_all_except(data_loader.unique_words)
    word_vectors.flatten()
    word_vectors.generate_missing_vectors(data_loader.unique_words, word_vectors_0)
    word_vectors.square()

    batch_size = 256

    text_writer = file_op.CSV_Writer("data/compositional_analysis/train/k_e/mult.csv", header=("k_e", "label"), delimiter=',')

    num_iters = len(data_loader) // batch_size
    last_batch_size = len(data_loader) - batch_size * num_iters - 1

    batch_sizes = [batch_size for _ in range(num_iters)] + [last_batch_size]

    for batch_size in batch_sizes:
        k_e = calc_ke_multiply(batch_size, data_loader, word_vectors)
        text_writer.append_lines(k_e)


def calc_ke_multiply(batch_size, data_loader, word_vectors):
    batch = data_loader.load_sequential(batch_size).to_model_data(["sentence1_parse", "sentence2_parse", "gold_label"])

    batch_1 = [find_all_pos_tags(sentence[0]) for sentence in batch]
    batch_1 = [[pair.lower().split(' ') for pair in sentence] for sentence in batch_1]

    batch_2 = [find_all_pos_tags(sentence[1]) for sentence in batch]
    batch_2 = [[pair.lower().split(' ') for pair in sentence] for sentence in batch_2]

    vectors_batch1 = efficient_vectors_from_batch(batch_1, word_vectors)
    vectors_batch2 = efficient_vectors_from_batch(batch_2, word_vectors)

    labels = [sentence[2] for sentence in batch]

    k_e = [[str(calc_k_e(vectors1, vectors2)), label]
           for vectors1, vectors2, label in zip(vectors_batch1, vectors_batch2, labels)]
    return k_e


def calc_k_e(sentence1_vectors, sentence2_vectors):
    try:
        vectors1 = multiply_sentence(sentence1_vectors)
        vectors2 = multiply_sentence(sentence2_vectors)
    except ValueError:
        return 0

    return hl.k_e(vectors1, vectors2)


def safe_lookup(word, word_vectors):
    vector = word_vectors.lookup(word)

    if np.all(vector == 0):
        return None

    return vector


def efficient_vectors_from_batch(batch, word_vectors):
    vectors = [[safe_lookup(pair[1], word_vectors)
                  for pair in sentence if safe_lookup(pair[1], word_vectors) is not None]
                 for sentence in batch]

    return vectors


def multiply_sentence(vectors):
    if not vectors:
        print("EMPTY VECTORS!")
        raise ValueError

    out = vectors[0]

    for vector in vectors[1:]:
        out = hl.mult(out, vector)

    return out


if __name__ == "__main__":
    ke_multiply("data/snli_1.0/snli_1.0_train.jsonl")
