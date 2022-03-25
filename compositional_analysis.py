import re

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from typing import List

import NLI_hyponomy_analysis.data_pipeline.file_operations as file_op
import NLI_hyponomy_analysis.data_pipeline.matrix_operations.hyponymy_library as hl
from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_Unclean, SentenceBatch
from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices, Hyponyms
from NLI_hyponomy_analysis.comp_analysis_library.parse_tree import ParseTree
from NLI_hyponomy_analysis.comp_analysis_library.policies import Policy, verbs_switch, only_mult

from sklearn.metrics import roc_curve, roc_auc_score

import os
import sys


label_mapping = {"t": 1, "entailment": 1, "neutral": 0.5, "f": 0, "contradiction": 0, '-': 0.5}


def get_test_words(word_sim_dir_path: str) -> List[str]:
    words = []
    for i, filename in enumerate(os.listdir(word_sim_dir_path)):
        for line in open(os.path.join(word_sim_dir_path, filename), 'r'):
            line = line.strip().lower()
            word1, word2, _ = line.split()
            words.extend([word1, word2])
    return words


def scatter(data_path: str, threshold=0.5):
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


def area_under_roc_curve(data_path: str, policy_name) -> float:
    predictions_loader = file_op.CSV_Writer(data_path, header="$auto", delimiter=',')
    __all_predictions = predictions_loader.load_all()

    predictions = np.array([float(prediction[0]) for prediction in __all_predictions])
    class_labels = np.array([label_mapping.get(prediction[1].lower(), 0.5) for prediction in __all_predictions])

    predictions = [predictions[i] for i in range(len(class_labels)) if class_labels[i] != 0.5]
    class_labels = [class_labels[i] for i in range(len(class_labels)) if class_labels[i] != 0.5]
    del __all_predictions

    fpr, tpr, thresholds = roc_curve(class_labels, predictions)
    auc = roc_auc_score(class_labels, predictions)
    print("OPTIMAL THRESHOLD:", thresholds[np.argmax(tpr - fpr)])
    auc_message = f"Area under curve: {round(auc, 8)}"

    fig_save_path = file_op.file_path_without_extension(data_path) + ".png"
    fig_save_path_info = file_op.child_path(data_path, 2)

    dataset = file_op.root_dir(fig_save_path_info)
    secondary_info = file_op.root_dir(file_op.child_path(fig_save_path_info, 1))
    plt.title(f"ROC curve for {dataset}, {secondary_info.upper()}, {policy_name.upper()}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr, tpr, linestyle='--', label=auc_message)
    plt.legend()
    plt.savefig(fig_save_path)
    plt.show()


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

    result = hl.k_e(vector1, vector2)
    return result


def k_a_from_two_vectors(vector1, vector2) -> float:
    if vector1 is None or vector2 is None:
        return None

    result = hl.k_ba(vector1, vector2, tol=1e-6)
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


def snli_stats(data_loader, word_vectors, policy, batch_size: int=256):
    batch = data_loader.load_sequential(batch_size).to_model_data(["sentence1_parse", "sentence2_parse", "gold_label"])
    k_e, k_a = __batch_stats(batch, word_vectors, policy, constructor=ParseTree)
    return k_e, k_a


def ks_stats(data_loader, word_vectors, policy, tags, batch_size: int=256):
    batch = data_loader.load_sequential(batch_size)

    def constructor(*args):
        return ParseTree.from_sentence(*args, tags=tags)
    k_e, k_a = __batch_stats(batch, word_vectors, policy, tags, constructor=constructor)
    return k_e, k_a


def __batch_stats(batch, word_vectors, policy: Policy, constructor=ParseTree):
    batch_1 = [sentence[0] for sentence in batch]
    batch_1 = [constructor(sentence, word_vectors, policy) for sentence in batch_1]
    for parse_tree in batch_1:
        parse_tree.evaluate()

    batch_2 = [sentence[1] for sentence in batch]
    batch_2 = [constructor(sentence, word_vectors, policy) for sentence in batch_2]
    for parse_tree in batch_2:
        parse_tree.evaluate()

    labels = [sentence[2] for sentence in batch]

    k_e = [[k_e_from_two_vectors(tree1.data[0], tree2.data[0]), label]
           for tree1, tree2, label in zip(batch_1, batch_2, labels)]
    k_e = [[str(line[0]), line[1]] for line in k_e if line[0] is not None]

    k_a = [[k_a_from_two_vectors(tree1.data[0], tree2.data[0]), label]
           for tree1, tree2, label in zip(batch_1, batch_2, labels)]
    k_a = [[str(line[0]), line[1]] for line in k_a if line[0] is not None]
    return k_e, k_a


def snli_test_policy(data_path: str, policy: Policy, policy_name, batch_size: int=256):
    load_dotenv()  # Path to the glove data directory -> HOME="..."
    data_loader = SNLI_DataLoader_Unclean(data_path)

    word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()
    word_vectors_0.remove_all_except(data_loader.unique_words)

    hyponyms = Hyponyms("data/hyponyms/25d_hyponyms_all.json", data_loader.unique_words)

    word_vectors = DenseHyponymMatrices(hyponyms, word_vectors_0.dict)

    data_file_path_k_e = f"data/compositional_analysis/train/k_e/{policy_name}"

    data_writer_k_e = file_op.CSV_Writer(data_file_path_k_e, header=("k_e", "label"),
                                         delimiter=',')

    data_file_path_k_a = f"data/compositional_analysis/train/k_a/{policy_name}"

    data_writer_k_a = file_op.CSV_Writer(data_file_path_k_a, header=("k_a", "label"),
                                         delimiter=',')

    if data_writer_k_e.file_exists or data_writer_k_a.file_exists:
        user_response = input("Do you want to continue, file already exists! ")
        if not user_response.lower() in ("y", "yes", "yh"):
            raise FileExistsError

    file_op.make_empty_file(data_file_path_k_e)
    file_op.make_empty_file(data_file_path_k_a)

    num_iters = len(data_loader) // batch_size
    last_batch_size = len(data_loader) - batch_size * num_iters - 1

    batch_sizes = [batch_size for _ in range(num_iters)] + [last_batch_size]

    for batch_size in batch_sizes:
        k_e, k_a = snli_test_policy(data_loader, word_vectors, policy, batch_size)
        data_writer_k_e.append_lines(k_e)
        data_writer_k_a.append_lines(k_a)


def test_ks(data_path: str, policy: Policy, policy_name: str, batch_size=256):
    load_dotenv()  # Path to the glove data directory -> HOME="..."

    ks_type = re.findall(r'[^\-]*$', file_op.file_path_without_extension(data_path))[0].lower()
    ks_type_to_tags = {"sv": ('n', 'v'), "vo": ('v', 'n'), "svo": ('n', 'v', 'n')}

    tags = ks_type_to_tags[ks_type]

    data_loader = file_op.CSV_Writer(data_path, delimiter=',')

    sentences = data_loader.load_all()
    sentences0 = SentenceBatch([' '.join(sentence[0:1]).lower() for sentence in sentences])

    word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()
    word_vectors_0.remove_all_except(sentences0.unique_words)

    hyponyms = Hyponyms("data/hyponyms/25d_hyponyms_all.json", sentences0.unique_words)

    word_vectors = DenseHyponymMatrices(hyponyms, word_vectors_0.dict)

    data_file_path_k_e = f"data/compositional_analysis/KS2016/{ks_type}/{policy_name}/k_e.csv"

    data_writer_k_e = file_op.CSV_Writer(data_file_path_k_e,
                                         header=("k_e", "label"),
                                         delimiter=',')

    data_file_path_k_a = f"data/compositional_analysis/KS2016/{ks_type}/{policy_name}/k_a.csv"

    data_writer_k_a = file_op.CSV_Writer(data_file_path_k_a,
                                         header=("k_e", "label"),
                                         delimiter=',')

    if data_writer_k_e.file_exists or data_writer_k_a.file_exists:
        user_response = input("Do you want to continue, file already exists! ")
        if not user_response.lower() in ("y", "yes", "yh"):
            raise FileExistsError

    dir_path = file_op.parent_path(data_file_path_k_e) + '/'
    file_op.make_dir(dir_path=dir_path)
    file_op.make_empty_file(data_file_path_k_e)

    dir_path = file_op.parent_path(data_file_path_k_a) + '/'
    file_op.make_dir(dir_path=dir_path)
    file_op.make_empty_file(data_file_path_k_a)

    num_iters = len(data_loader) // batch_size
    last_batch_size = len(data_loader) - batch_size * num_iters - 1

    batch_sizes = [batch_size for _ in range(num_iters)] + [last_batch_size]

    for batch_size in batch_sizes:
        k_e, k_a = ks_stats(data_loader, word_vectors, policy, tags, batch_size=batch_size)
        data_writer_k_e.append_lines(k_e)
        data_writer_k_a.append_lines(k_e)


def testing(data_path: str, depth: int=10):
    load_dotenv()  # Path to the glove data directory -> HOME="..."

    word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()

    words = get_test_words("data/word-sim")
    word_vectors_0.remove_all_except(words)

    unique_words = word_vectors_0.words
    vectors = word_vectors_0.dict

    hyponyms_all = Hyponyms(f"data/hyponyms/depth_25/hyps_depth_{depth}.json", unique_words, depth=depth)
    word_vectors = DenseHyponymMatrices(hyponyms=hyponyms_all, embedding_vectors=vectors)

    word_vectors.flatten()
    word_vectors.to_csv(data_path)


def testing2(data_path: str):
    load_dotenv()  # Path to the glove data directory -> HOME="..."
    data_loader = SNLI_DataLoader_Unclean(data_path)

    word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()
    word_vectors_0.remove_all_except(list(set([word.lower() for word in data_loader.unique_words])))

    hyponyms = Hyponyms("data/hyponyms/25d_hyponyms_all.json", data_loader.unique_words)

    word_vectors = DenseHyponymMatrices(hyponyms, word_vectors_0.dict)
    word_vectors.flatten()

    print(sys.getsizeof(data_loader.load_sequential(5).to_model_data().to_tensors(word_vectors)))

    #
    # k_e = snli_test_policy(data_loader, word_vectors, 1)
    # print(k_e)


def ks_test_subset(subset, policy: Policy, policy_name: str):
    test_ks(f"data/KS2016/KS2016-{subset.upper()}.csv", policy, policy_name)
    area_under_roc_curve(f"data/compositional_analysis/KS2016/{subset}/{policy_name}/k_e.csv", policy_name)
    scatter(f"data/compositional_analysis/KS2016/{subset}/{policy_name}/k_e.csv")


def ks_test_policy(policy: Policy, policy_name: str):
    subsets = ("sv", "vo", "svo")

    for subset in subsets:
        ks_test_subset(subset, policy, policy_name)


def main():
    # testing2("data/snli_1.0/snli_1.0_train.jsonl")

    depths = [1, 2, 4, 6, 8, 10, 12, 15, 20]

    for depth in depths:
        testing(f"data/word_sims_vectors/how_depth_affects_glove_25/depth_{depth}.csv", depth=depth)

    # policy = only_mult()
    # snli_test_policy()
    # policy = verbs_switch()
    #
    # ks_test_policy(policy, policy_name="verbs_switch")


if __name__ == "__main__":
    main()
