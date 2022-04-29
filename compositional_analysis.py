import re

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from typing import List

import NLI_hyponomy_analysis.data_pipeline.file_operations as file_op
import NLI_hyponomy_analysis.data_pipeline.word_operations as word_op
import NLI_hyponomy_analysis.data_pipeline.matrix_operations.hyponymy_library as hl
from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_Unclean, SentenceBatch
from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices, Hyponyms
from NLI_hyponomy_analysis.comp_analysis_library.parse_tree import ParseTree
from NLI_hyponomy_analysis.comp_analysis_library.policies import Policy, only_mult_trace

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


def scatter(data_path: str, fig_type="k_e"):
    fig_save_path = file_op.file_path_without_extension(data_path) + "_SCATTER.png"
    if file_op.is_file(fig_save_path):
        return None

    data_loader = file_op.CSV_Writer(data_path, delimiter=',', header="$auto")
    data = data_loader.load_as_dataframe()

    values = data[fig_type].to_list()
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

        if fig_type == "k_a":
            value = (value + 1) / 2

        point_to_add = value

        points[i, :] = point_to_add, label_encoding
        colors[i] = plot_color

    n = len(values)
    print(f"Percentage accuracy {round(number_correct / n, 2)} from a sample of {n} sentences.")

    plt.title(f"Distribution of sentences for training, under {fig_type.capitalize()}, mult.")
    plt.xlabel(f"Predicted Value, {fig_type.capitalize()}")
    plt.ylabel("Actual Value")

    plt.scatter(points[:, 0], points[:, 1], c=colors, s=1)

    plt.savefig(fig_save_path)
    plt.close()


def area_under_roc_curve(data_path: str, policy_name):
    fig_save_path = file_op.file_path_without_extension(data_path) + "_AUC.png"

    if file_op.is_file(fig_save_path):
        return None

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

    fig_save_path_info = file_op.child_path(data_path, 2)

    dataset = file_op.root_dir(fig_save_path_info)
    secondary_info = file_op.root_dir(file_op.child_path(fig_save_path_info, 1))
    plt.title(f"ROC curve for {dataset}, {secondary_info.upper()}, {policy_name.upper()}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr, tpr, linestyle='--', label=auc_message)
    plt.legend()
    plt.savefig(fig_save_path)
    plt.close()


def plot_stats(plot_dir: str, policy_name) -> None:
    assert plot_dir[-1] == "/", TypeError
    k_e_path = plot_dir + 'k_e.csv'
    k_a_path = plot_dir + 'k_a.csv'
    area_under_roc_curve(k_e_path, policy_name)
    area_under_roc_curve(k_a_path, policy_name)
    scatter(k_e_path, fig_type="k_e")
    scatter(k_a_path, fig_type="k_a")
    return None


def k_e_from_batches(sentence1_vectors, sentence2_vectors):
    try:
        vectors1 = hadamard_list_of_vectors(sentence1_vectors)
        vectors2 = hadamard_list_of_vectors(sentence2_vectors)
    except ValueError:
        return 0

    try:
        result = hl.k_e(vectors1, vectors2)
    except np.linalg.LinAlgError:
        return 0.5  # Prediction for neutral?

    return result


def k_e_from_two_vectors(vector1, vector2) -> float:
    if vector1 is None or vector2 is None:
        return 0.0

    try:
        result = hl.k_e(vector1, vector2)
    except np.linalg.LinAlgError:
        return 0
    if np.isnan(result):
        return 0
    return result


def k_a_from_two_vectors(vector1, vector2) -> float:
    if vector1 is None or vector2 is None:
        return 0

    try:
        result = hl.k_ba(vector1, vector2, tol=1e-6)
    except np.linalg.LinAlgError:
        return 0  # Prediction for neutral?
    if np.isnan(result):
        return 0

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

    k_e, k_a = __batch_stats(batch, word_vectors, policy, constructor=constructor)
    return k_e, k_a


def __batch_stats(batch, word_vectors, policy: Policy, constructor=ParseTree):
    batch_1 = [sentence[0] for sentence in batch]
    batch_1 = [constructor(word_op.remove_utf8_bad_chars(sentence), word_vectors, policy) for sentence in batch_1]
    for parse_tree in batch_1:
        parse_tree.evaluate()

    batch_2 = [sentence[1] for sentence in batch]
    batch_2 = [constructor(word_op.remove_utf8_bad_chars(sentence), word_vectors, policy) for sentence in batch_2]
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


def nli_test_policy(data_loader: SNLI_DataLoader_Unclean, data_name: str, policy: Policy, policy_name, batch_size: int=256):
    load_dotenv()  # Path to the glove data directory -> HOME="..."

    word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()
    word_vectors_0.remove_all_except(data_loader.unique_words)

    hyponyms = Hyponyms("data/hyponyms/25d_hyponyms_all.json", data_loader.unique_words)

    word_vectors = DenseHyponymMatrices(hyponyms, word_vectors_0.dict)

    data_file_path_k_e = f"data/compositional_analysis/{policy_name}/{data_name}/k_e.csv"

    data_writer_k_e = file_op.CSV_Writer(data_file_path_k_e, header=("k_e", "label"),
                                         delimiter=',')

    data_file_path_k_a = f"data/compositional_analysis/{policy_name}/{data_name}/k_a.csv"

    data_writer_k_a = file_op.CSV_Writer(data_file_path_k_a, header=("k_a", "label"),
                                         delimiter=',')

    if data_writer_k_e.file_exists and data_writer_k_a.file_exists:
        plot_stats(f"data/compositional_analysis/{policy_name}/{data_name}/", policy_name)
        return None

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
        k_e, k_a = snli_stats(data_loader, word_vectors, policy, batch_size)
        data_writer_k_e.append_lines(k_e)
        data_writer_k_a.append_lines(k_a)

    plot_stats(f"data/compositional_analysis/{policy_name}/{data_name}/", policy_name)


def ks_test_policy(data_path: str, policy: Policy, policy_name: str, batch_size=256):
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

    data_file_path_k_e = f"data/compositional_analysis/{policy_name}/KS2016/{ks_type}/k_e.csv"

    data_writer_k_e = file_op.CSV_Writer(data_file_path_k_e,
                                         header=("k_e", "label"),
                                         delimiter=',')

    data_file_path_k_a = f"data/compositional_analysis/{policy_name}/KS2016/{ks_type}/k_a.csv"

    data_writer_k_a = file_op.CSV_Writer(data_file_path_k_a,
                                         header=("k_a", "label"),
                                         delimiter=',')

    if data_writer_k_e.file_exists and data_writer_k_a.file_exists:
        plot_stats(f"data/compositional_analysis/{policy_name}/KS2016/{ks_type}/", policy_name)
        return None

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

    plot_stats(f"data/compositional_analysis/{policy_name}/KS2016/{ks_type}/", policy_name)


def test_depth(data_path: str, depth: int=10):
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


def ks_test_policy_all(policy: Policy, policy_name: str):
    subsets = ("sv", "vo", "svo")

    for subset in subsets:
        ks_test_policy(f"data/KS2016/KS2016-{subset.upper()}.csv", policy, policy_name)


def sick_test_policy_all(policy: Policy, policy_name: str):
    data_loader = file_op.CSV_Writer(f"data/SICK/SICK_annotated.csv", delimiter='\t', header="$auto")
    data_all = data_loader.load_as_dataframe()
    data_all_all = data_all["entailment_label"]

    data_train = data_all[data_all["SemEval_set"] == "TRAIN"]
    data_test = data_all[data_all["SemEval_set"] == "TEST"]

    # sick_test_policy(data_loader, )


def test_policy(policy: Policy, policy_name: str):
    ks_test_policy_all(policy, policy_name=policy_name)

    snli_data_loader = SNLI_DataLoader_Unclean(f"data/snli_1.0/snli_1.0_test.jsonl")
    nli_test_policy(snli_data_loader, "SNLI", policy, policy_name)

    mnli_data_loader = SNLI_DataLoader_Unclean(f"data/mnli/multinli_1.0/multinli_1.0_dev_matched.jsonl")
    nli_test_policy(mnli_data_loader, "MNLI", policy, policy_name)


def main():
    depths = [1, 2, 4, 6, 8, 10, 12, 15, 20]

    for depth in depths:
        test_depth(f"data/word_sims_vectors/how_depth_affects_glove_25/depth_{depth}.csv", depth=depth)


if __name__ == "__main__":
    _policy = only_mult_trace()

    test_policy(_policy, policy_name="mult_only_norm2")
