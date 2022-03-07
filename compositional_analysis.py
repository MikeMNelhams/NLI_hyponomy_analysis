import re

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from typing import List

import NLI_hyponomy_analysis.data_pipeline.file_operations as file_op
import NLI_hyponomy_analysis.data_pipeline.matrix_operations.hyponymy_library as hl
from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_Unclean, SentenceBatch
from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices2, Hyponyms
from NLI_hyponomy_analysis.data_pipeline.word_operations import find_all_pos_tags
from parse_tree import ParseTree

from sklearn.metrics import roc_curve, roc_auc_score

import os


label_mapping = {"t": 1, "entailment": 1, "neutral": 0.5, "f": 0, "contradiction": 0, '-': 0.5}


def get_test_words(word_sim_dir_path: str) -> List[str]:
    words = []
    for i, filename in enumerate(os.listdir(word_sim_dir_path)):
        for line in open(os.path.join(word_sim_dir_path, filename), 'r'):
            line = line.strip().lower()
            word1, word2, _ = line.split()
            words.extend([word1, word2])
    return words


def area_under_roc_curve(data_path: str) -> float:
    predictions_loader = file_op.CSV_Writer(data_path, header="$auto", delimiter=',')
    __all_predictions = predictions_loader.load_all()

    predictions = np.array([float(prediction[0]) for prediction in __all_predictions])
    class_labels = np.array([label_mapping[prediction[1].lower()] for prediction in __all_predictions])
    del __all_predictions

    print(class_labels)
    print(predictions)
    fpr, tpr, thresholds = roc_curve(class_labels, predictions)
    auc = roc_auc_score(class_labels, predictions)

    auc_message = f"Area under curve: {auc}"

    print(auc_message)
    plt.title(f"ROC curve for {data_path}")
    plt.plot(fpr, tpr, linestyle='--', label=data_path)
    plt.show()


def test_snli_mult(data_path: str):
    load_dotenv()  # Path to the glove data directory -> HOME="..."

    data_loader = SNLI_DataLoader_Unclean(data_path)

    word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()
    word_vectors_0.remove_all_except(data_loader.unique_words)

    hyponyms = Hyponyms("data/hyponyms/25d_hyponyms_all.json")

    word_vectors = DenseHyponymMatrices2(hyponyms, word_vectors_0.dict)
    word_vectors.flatten()
    word_vectors.square()

    batch_size = 256

    data_writer = file_op.CSV_Writer("data/compositional_analysis/train/k_e/mult.csv", header=("k_e", "label"), delimiter=',')
    if data_writer.file_exists:
        raise FileExistsError

    num_iters = len(data_loader) // batch_size
    last_batch_size = len(data_loader) - batch_size * num_iters - 1

    batch_sizes = [batch_size for _ in range(num_iters)] + [last_batch_size]

    for batch_size in batch_sizes:
        k_e = calc_ke_multiply(batch_size, data_loader, word_vectors)
        data_writer.append_lines(k_e)


def calc_ke_multiply(batch_size, data_loader, word_vectors):
    batch = data_loader.load_sequential(batch_size).to_model_data(["sentence1_parse", "sentence2_parse", "gold_label"])

    batch_1 = [find_all_pos_tags(sentence[0]) for sentence in batch]
    batch_1 = [[pair.lower().split(' ') for pair in sentence] for sentence in batch_1]

    batch_2 = [find_all_pos_tags(sentence[1]) for sentence in batch]
    batch_2 = [[pair.lower().split(' ') for pair in sentence] for sentence in batch_2]

    vectors_batch1 = efficient_vectors_from_batch(batch_1, word_vectors)
    vectors_batch2 = efficient_vectors_from_batch(batch_2, word_vectors)

    labels = [sentence[2] for sentence in batch]

    k_e = [[str(hl.k_e(vectors1, vectors2)), label]
           for vectors1, vectors2, label in zip(vectors_batch1, vectors_batch2, labels)]
    return k_e


def k_e_from_batches(sentence1_vectors, sentence2_vectors):
    try:
        vectors1 = hadamard_list_of_vectors(sentence1_vectors)
        vectors2 = hadamard_list_of_vectors(sentence2_vectors)
    except ValueError:
        return 0

    return hl.k_e(vectors1, vectors2)


def k_e_from_two_vectors(vector1, vector2) -> float:
    if vector1 is None or vector2 is None:
        return None
    result = 0
    try:
        result = hl.k_e(vector1[0], vector2[0])
    except:
        print(f"Failed for vectors: \'{vector1}\' and \'{vector2}\'")
    return result


def efficient_vectors_from_batch(batch, word_vectors):
    vectors = [[word_vectors.safe_lookup(pair[1])
                  for pair in sentence if word_vectors.safe_lookup(pair[1]) is not None]
                 for sentence in batch]

    return vectors


def hadamard_list_of_vectors(vectors):
    if not vectors:
        print("EMPTY VECTORS!")
        raise ValueError

    out = vectors[0]

    for vector in vectors[1:]:
        out = hl.mult(out, vector)

    return out


def snli_pos_k_e(data_loader, word_vectors, batch_size: int=256):
    batch = data_loader.load_sequential(batch_size).to_model_data(["sentence1_parse", "sentence2_parse", "gold_label"])

    batch_1 = [sentence[0] for sentence in batch]
    batch_1 = [ParseTree(sentence, word_vectors) for sentence in batch_1]

    for parse_tree in batch_1:
        parse_tree.evaluate()

    batch_2 = [sentence[1] for sentence in batch]
    batch_2 = [ParseTree(sentence, word_vectors) for sentence in batch_2]
    for parse_tree in batch_2:
        parse_tree.evaluate()

    labels = [sentence[2] for sentence in batch]

    k_e = [[k_e_from_two_vectors(tree1.data[0], tree2.data[0]), label]
           for tree1, tree2, label in zip(batch_1, batch_2, labels)]
    k_e = [[str(line[0]), line[1]] for line in k_e if line[0] is not None]
    return k_e


def ks2016_pos_k_e(data_loader, word_vectors, batch_size: int=256, tags=None):
    batch = data_loader.load_sequential(batch_size)

    batch_1 = [sentence[0] for sentence in batch]
    batch_1 = [ParseTree.from_untagged_sentence(sentence, word_vectors, tags=tags) for sentence in batch_1]

    for parse_tree in batch_1:
        parse_tree.evaluate()

    batch_2 = [sentence[1] for sentence in batch]
    batch_2 = [ParseTree.from_untagged_sentence(sentence, word_vectors, tags=tags) for sentence in batch_2]
    for parse_tree in batch_2:
        parse_tree.evaluate()

    labels = [sentence[2] for sentence in batch]

    k_e = [[k_e_from_two_vectors(tree1.data[0], tree2.data[0]), label]
           for tree1, tree2, label in zip(batch_1, batch_2, labels)]
    k_e = [[str(line[0]), line[1]] for line in k_e if line[0] is not None]
    return k_e


def test_snli(data_path: str, batch_size: int=256):
    load_dotenv()  # Path to the glove data directory -> HOME="..."
    data_loader = SNLI_DataLoader_Unclean(data_path)

    word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()
    word_vectors_0.remove_all_except(data_loader.unique_words)

    hyponyms = Hyponyms("data/hyponyms/25d_hyponyms_all.json", data_loader.unique_words)

    word_vectors = DenseHyponymMatrices2(hyponyms, word_vectors_0.dict)
    word_vectors.flatten()
    word_vectors.square()

    data_writer = file_op.CSV_Writer("data/compositional_analysis/train/k_e/pos_tree2.csv", header=("k_e", "label"),
                                     delimiter=',')
    # if data_writer.file_exists:
    #     raise FileExistsError

    num_iters = len(data_loader) // batch_size
    last_batch_size = len(data_loader) - batch_size * num_iters - 1

    batch_sizes = [batch_size for _ in range(num_iters)] + [last_batch_size]

    for batch_size in batch_sizes:
        k_e = snli_pos_k_e(data_loader, word_vectors, batch_size)
        data_writer.append_lines(k_e)


def test_ks2016(data_path: str, tags_enabled=True):
    load_dotenv()  # Path to the glove data directory -> HOME="..."

    ks_type = re.findall(r'[^\-]*$', file_op.file_path_without_extension(data_path))[0].lower()
    ks_type_to_tags = {"sv": ('n', 'v'), "vo": ('v', 'n'), "svo": ('n', 'v', 'n')}
    tags = None
    if tags_enabled and ks_type_to_tags is not None:
        tags = ks_type_to_tags[ks_type]

    data_loader = file_op.CSV_Writer(data_path, delimiter=',')

    sentences = data_loader.load_all()
    sentences0 = SentenceBatch([' '.join(sentence[0:1]).lower() for sentence in sentences])

    word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()
    word_vectors_0.remove_all_except(sentences0.unique_words)

    hyponyms = Hyponyms("data/hyponyms/25d_hyponyms_all.json", sentences0.unique_words)

    word_vectors = DenseHyponymMatrices2(hyponyms, word_vectors_0.dict)
    word_vectors.flatten()
    word_vectors.square()

    data_writer = file_op.CSV_Writer(f"data/compositional_analysis/KS2016/{ks_type}/k_e/pos_tree2.csv",
                                     header=("k_e", "label"),
                                     delimiter=',')
    batch_size = 256

    if data_writer.file_exists:
        raise FileExistsError

    num_iters = len(data_loader) // batch_size
    last_batch_size = len(data_loader) - batch_size * num_iters - 1

    batch_sizes = [batch_size for _ in range(num_iters)] + [last_batch_size]

    for batch_size in batch_sizes:
        k_e = ks2016_pos_k_e(data_loader, word_vectors, batch_size, tags=tags)
        data_writer.append_lines(k_e)


def scatter(data_path: str):
    data_loader = file_op.CSV_Writer(data_path, delimiter=',')

    data = data_loader.load_as_dataframe()

    values = data["k_e"].to_list()
    labels = data["label"].to_list()

    points = np.zeros((len(values), 2))
    colors = ["blue" for _ in range(len(values))]

    number_correct = 0

    for i, (value, label) in enumerate(zip(values, labels)):
        plot_color = 'blue'
        label_encoding = label_mapping[label.lower()]
        if label_encoding == 1:
            plot_color = 'green'
        elif label_encoding == 0.5:
            plot_color = 'blue'
        elif label_encoding == 0:
            plot_color = 'red'

        if round(value) == label_encoding:
            number_correct += 1

        point_to_add = round(value, 2)
        if abs(point_to_add) > 1 or point_to_add < 0:
            point_to_add = 0

        points[i, :] = point_to_add, label_encoding
        colors[i] = plot_color

    n = len(values)
    print(f"Percentage accuracy {round(number_correct / n, 2)} from a sample of {n} sentences.")

    plt.title("Distribution of sentences for training, under k_e, mult.")
    plt.xlabel("Predicted Value, K_e")
    plt.ylabel("Actual Value")

    plt.scatter(points[:, 0], points[:, 1], c=colors, s=1)
    plt.show()


def testing(data_path: str):
    load_dotenv()  # Path to the glove data directory -> HOME="..."

    word_vectors_0 = embed.Embedding2('common_crawl_840', d_emb=300, show_progress=True, default='zero')
    word_vectors_0.load_memory()

    words = get_test_words("data/word-sim")
    word_vectors_0.remove_all_except(words)

    unique_words = word_vectors_0.words
    vectors = word_vectors_0.dict

    hyponyms_all = Hyponyms("data/hyponyms/300cc840_hyponyms_wordsim.json", unique_words)
    word_vectors = DenseHyponymMatrices2(hyponyms=hyponyms_all, embedding_vectors=vectors)

    word_vectors.flatten()
    word_vectors.to_csv(data_path)


if __name__ == "__main__":
    # test_snli_mult("data/snli_1.0/snli_1.0_train.jsonl")
    # scatter("data/compositional_analysis/train/k_e/mult.csv")
    # test_snli("data/snli_1.0/snli_1.0_train.jsonl")
    # area_under_roc_curve("data/compositional_analysis/train/k_e/pos_tree2.csv")
    # scatter("data/compositional_analysis/train/k_e/pos_tree2.csv")
    # test_ks2016("data/KS2016/KS2016-SV.csv", tags_enabled=False)
    area_under_roc_curve("data/compositional_analysis/KS2016/sv/k_e/pos_tree2.csv")
    scatter("data/compositional_analysis/KS2016/sv/k_e/pos_tree2.csv")
    # testing("data/word_sims_vectors/300cc840_glove_hypo_wordsim.csv")

    # test_ks2016("data/KS2016/KS2016-VO.csv", tags_enabled=False)
    # area_under_roc_curve("data/compositional_analysis/KS2016/vo/k_e/pos_tree2.csv")
    # scatter("data/compositional_analysis/KS2016/vo/k_e/pos_tree2.csv")
