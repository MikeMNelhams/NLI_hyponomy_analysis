import re

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from typing import List, Any

import NLI_hyponomy_analysis.data_pipeline.file_operations as file_op
import NLI_hyponomy_analysis.data_pipeline.word_operations as word_op
import NLI_hyponomy_analysis.data_pipeline.matrix_operations.hyponymy_library as hl
from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_Unclean, SentenceBatch
from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices, Hyponyms
from NLI_hyponomy_analysis.comp_analysis_library.parse_tree import ParseTree
from NLI_hyponomy_analysis.comp_analysis_library.policies import Policy, only_addition, only_mult, verbs_switch, comp_policy1, example_policy, example_policy_no_scaling

import nltk

from sklearn.metrics import roc_curve, roc_auc_score

import os


label_mapping = {"t": 1, "entailment": 1, "neutral": 0.5, "f": 0, "contradiction": 0, '-': 0.5}
label_mapping_multi_class = {"entailment": 2, "neutral": 1, "contradiction": 0}
label_mapping_multi_class_vectors = {"entailment": [0, 0, 1], "neutral": [0, 1, 0], "contradiction": [1, 0, 0]}


def get_test_words(word_sim_dir_path: str) -> List[str]:
    words = []
    for i, filename in enumerate(os.listdir(word_sim_dir_path)):
        for line in open(os.path.join(word_sim_dir_path, filename), 'r'):
            line = line.strip().lower()
            word1, word2, _ = line.split()
            words.extend([word1, word2])
    return words


def make_k_file(data_file_path_k: str) -> None:
    dir_path = file_op.parent_path(data_file_path_k) + '/'
    file_op.make_dir(dir_path=dir_path)
    file_op.make_empty_file(data_file_path_k)
    return None


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


def load_predictions(data_path: str):
    predictions_loader = file_op.CSV_Writer(data_path, header="$auto", delimiter=',')
    __all_predictions = predictions_loader.load_all()

    # for prediction in __all_predictions:
    #     if prediction[1] == "contradiction" or prediction[1] == "neutral" or prediction[1] == "entailment":
    #         print("MULTI CLASS NOT CONVERTED!")
    #         raise ValueError

    return __all_predictions


def auc_graph_single_class(data, figure_save_path: str, policy_name: str, legend_data_name: str,
                           dataset_name: str, fig=None, axes=None, save: bool=False, linestyle='--') -> (plt.Figure, plt.Axes):
    """ Saving will close the figure."""
    fpr, tpr, thresholds, auc = auc_single_class(data)
    print("OPTIMAL THRESHOLD:", thresholds[np.argmax(tpr - fpr)])

    ax_plot = axes
    fig_plot = fig
    if fig is None and axes is not None:
        raise ValueError("Axes should not be passed without the Figure!")
    if axes is None:
        if fig is not None:
            raise ValueError("Figure should not be passed without the Axes!")
        fig_plot, ax_plot = plt.subplots()
        ax_plot.set_title(f"ROC curve for {dataset_name}, {policy_name.upper()}", fontsize=11)
        ax_plot.set_xlabel("False Positive Rate", fontsize=16)
        ax_plot.set_ylabel("True Positive Rate", fontsize=16)

    auc_message = f"{legend_data_name} - Area Under Curve: {round(auc, 6)}"
    ax_plot.plot(fpr, tpr, linestyle=linestyle, label=auc_message)

    if save:
        ax_plot.legend(fontsize=13)
        fig_plot.savefig(figure_save_path)
        plt.close()

    return fig_plot, ax_plot


def plot_auc_curve_single(dir_path: str, policy_name: str) -> None:
    fig_save_path = dir_path + "AUC.png"
    k_e_path = dir_path + "k_e.csv"
    k_a_path = dir_path + "k_a.csv"

    k_e_predictions = load_predictions(k_e_path)
    k_a_predictions = load_predictions(k_a_path)

    dataset_name = file_op.parent_path_name(k_e_path)
    if dataset_name.lower() in ("sv", "svo", "vo"):
        dataset_name = file_op.parent_path_name(file_op.parent_path(k_e_path)) + ' - ' + dataset_name.upper()

    print('-' * 50)
    print(f"Saving graph to {fig_save_path}")
    print('-' * 50)
    fig, axes = auc_graph_single_class(k_e_predictions, fig_save_path, policy_name, dataset_name=dataset_name, legend_data_name="k_e")
    auc_graph_single_class(k_a_predictions, fig_save_path, policy_name, dataset_name=dataset_name,
                           legend_data_name="k_a", fig=fig, axes=axes, linestyle='dashdot', save=True)
    return None


def plot_auc_curve_multi(dir_path: str, policy_name: str) -> None:
    fig_save_path = dir_path + "AUC.png"
    k_e_path = dir_path + "k_e.csv"
    k_a_path = dir_path + "k_a.csv"

    k_e_predictions = load_predictions(k_e_path)
    k_a_predictions = load_predictions(k_a_path)

    auc_k_e = auc_multi_class_k_e(k_e_predictions)
    raise ZeroDivisionError
    auc_k_a = auc_multi_class(k_a_predictions)


def auc_single_class(predictions_input) -> (Any, Any, Any, float):
    # Labels are 0 or 1, simple binary classification, but mutual, so kinda like a single class.
    predictions = np.array([float(prediction[0]) for prediction in predictions_input])
    labels = [label_mapping.get(prediction[1].lower(), 0.5) for prediction in predictions_input]
    labels = [label for label in labels if label != 0.5]
    class_labels = np.array(labels)
    predictions = [predictions[i] for i in range(len(class_labels)) if class_labels[i] != 0.5]
    class_labels = [class_labels[i] for i in range(len(class_labels)) if class_labels[i] != 0.5]

    fpr, tpr, thresholds = roc_curve(class_labels, predictions)
    auc = roc_auc_score(class_labels, predictions)
    return fpr, tpr, thresholds, auc


def auc_multi_class_k_e(predictions_input):
    # K_E ranges from 0 to 1
    # Compute ROC curve and ROC area for each class
    predictions = [float(prediction[0]) for prediction in predictions_input]

    # predictions = np.array([[1 - prediction, 0, prediction] for prediction in predictions])

    labels = [label_mapping_multi_class.get(prediction[1].lower(), -1) for prediction in predictions_input]
    labels = np.array([label for label in labels if label != -1])

    weighted_roc_auc_ovr = roc_auc_score(labels, predictions, multi_class="ovr", average="macro")

    return weighted_roc_auc_ovr


def plot_stats_single(plot_dir: str, policy_name) -> None:
    assert plot_dir[-1] == "/", TypeError
    plot_auc_curve_single(plot_dir, policy_name)

    k_e_path = plot_dir + 'k_e.csv'
    k_a_path = plot_dir + 'k_a.csv'
    scatter(k_e_path, fig_type="k_e")
    scatter(k_a_path, fig_type="k_a")
    return None


def plot_stats_multi(plot_dir: str, policy_name) -> None:
    assert plot_dir[-1] == "/", TypeError

    k_e_path = plot_dir + 'k_e.csv'
    k_a_path = plot_dir + 'k_a.csv'
    scatter(k_e_path, fig_type="k_e")
    scatter(k_a_path, fig_type="k_a")

    plot_auc_curve_multi(plot_dir, policy_name)
    return None


def k_e_from_two_trees(tree1: ParseTree, tree2: ParseTree) -> float:
    return tree1.metric(tree2, binary_metric=hl.k_e)


def k_a_from_two_trees(tree1: ParseTree, tree2: ParseTree) -> float:
    def metric(x, y) -> float:
        return hl.k_ba(x, y, tol=1e-6)

    return tree1.metric(tree2, binary_metric=metric)


def efficient_vectors_from_batch(batch, word_vectors):
    vectors = [[word_vectors.safe_lookup(pair[1])
                  for pair in sentence if word_vectors.safe_lookup(pair[1]) is not None]
                 for sentence in batch]
    return vectors


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

    k_e = [[k_e_from_two_trees(tree1, tree2), label]
           for tree1, tree2, label in zip(batch_1, batch_2, labels)]
    k_e = [[str(line[0]), line[1]] for line in k_e if line[0] is not None]

    k_a = [[k_a_from_two_trees(tree1, tree2), label]
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
    data_writer_k_e = file_op.CSV_Writer(data_file_path_k_e, header=("k_e", "label"), delimiter=',')

    data_file_path_k_a = f"data/compositional_analysis/{policy_name}/{data_name}/k_a.csv"
    data_writer_k_a = file_op.CSV_Writer(data_file_path_k_a, header=("k_a", "label"), delimiter=',')

    if data_writer_k_e.file_exists and data_writer_k_a.file_exists:
        plot_stats_multi(f"data/compositional_analysis/{policy_name}/{data_name}/", policy_name)
        return None

    make_k_file(data_file_path_k_e)
    make_k_file(data_file_path_k_a)

    num_iters = len(data_loader) // batch_size
    last_batch_size = len(data_loader) - batch_size * num_iters - 1

    batch_sizes = [batch_size for _ in range(num_iters)] + [last_batch_size]

    for batch_size in batch_sizes:
        k_e, k_a = snli_stats(data_loader, word_vectors, policy, batch_size)
        data_writer_k_e.append_lines(k_e)
        data_writer_k_a.append_lines(k_a)

    plot_stats_multi(f"data/compositional_analysis/{policy_name}/{data_name}/", policy_name)


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
    data_writer_k_e = file_op.CSV_Writer(data_file_path_k_e, header=("k_e", "label"), delimiter=',')

    data_file_path_k_a = f"data/compositional_analysis/{policy_name}/KS2016/{ks_type}/k_a.csv"
    data_writer_k_a = file_op.CSV_Writer(data_file_path_k_a, header=("k_a", "label"), delimiter=',')

    if data_writer_k_e.file_exists and data_writer_k_a.file_exists:
        plot_stats_single(f"data/compositional_analysis/{policy_name}/KS2016/{ks_type}/", policy_name)
        return None

    make_k_file(data_file_path_k_e)
    make_k_file(data_file_path_k_a)

    num_iters = len(data_loader) // batch_size
    last_batch_size = len(data_loader) - batch_size * num_iters - 1
    batch_sizes = [batch_size for _ in range(num_iters)] + [last_batch_size]

    for batch_size in batch_sizes:
        k_e, k_a = ks_stats(data_loader, word_vectors, policy, tags, batch_size=batch_size)
        data_writer_k_e.append_lines(k_e)
        data_writer_k_a.append_lines(k_a)

    plot_stats_single(f"data/compositional_analysis/{policy_name}/KS2016/{ks_type}/", policy_name)
    return None


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
    data_all_labels = data_all["entailment_label"]

    data_train = data_all[data_all["SemEval_set"] == "TRAIN"]
    data_test = data_all[data_all["SemEval_set"] == "TEST"]
    print(data_all.columns)

    # sick_test_policy(data_loader, )


def test_policy(policy: Policy, policy_name: str):
    # ks_test_policy_all(policy, policy_name=policy_name)
    load_dotenv()  # Path to the glove data directory -> HOME="..."

    words = ["man", "plays", "piano", "instrument", "runs", "around", "is", "musician", "noise"]

    word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()

    hyponyms = Hyponyms("data/hyponyms/25d_hyponyms_all.json", words)
    word_vectors = DenseHyponymMatrices(hyponyms, word_vectors_0.dict)

    sentence1 = "(ROOT (S (NP man) (VBZ plays) (NN piano)))"

    sentence2 = "(ROOT (S (NP man) (VBZ plays) (NN instrument)))"
    sentence3 = "(ROOT (S (NP man) (VBZ makes) (NN noise)))"
    sentence4 = "(ROOT (S (NP man) (VBZ runs) (NN around)))"
    sentence5 = "(ROOT (S (NP man) (VBZ is) (NN musician)))"

    tree1 = ParseTree(sentence1, word_vectors, policy)
    tree2 = ParseTree(sentence2, word_vectors, policy)
    tree3 = ParseTree(sentence3, word_vectors, policy)
    tree4 = ParseTree(sentence4, word_vectors, policy)
    tree5 = ParseTree(sentence5, word_vectors, policy)

    tree1.evaluate()
    tree2.evaluate()
    tree3.evaluate()
    tree4.evaluate()
    tree5.evaluate()

    k_e_res1 = tree1.metric(tree2, hl.k_e)
    k_a_res1 = tree1.metric(tree2, hl.k_ba)

    k_e_res2 = tree1.metric(tree3, hl.k_e)
    k_a_res2 = tree1.metric(tree3, hl.k_ba)

    k_e_res3 = tree1.metric(tree4, hl.k_e)
    k_a_res3 = tree1.metric(tree4, hl.k_ba)

    k_e_res4 = tree1.metric(tree5, hl.k_e)
    k_a_res4 = tree1.metric(tree5, hl.k_ba)

    print(f"1 and 2: k_e {k_e_res1} k_a {k_a_res1}")
    print(f"1 and 3: k_e {k_e_res2} k_a {k_a_res2}")
    print(f"1 and 4: k_e {k_e_res3} k_a {k_a_res3}")
    print(f"1 and 5: k_e {k_e_res4} k_a {k_a_res4}")


def depth_shifting():
    depths = [1, 2, 4, 6, 8, 10, 12, 15, 20]

    for depth in depths:
        test_depth(f"data/word_sims_vectors/how_depth_affects_glove_25/depth_{depth}.csv", depth=depth)


def test_policy_all(policy: Policy, policy_name: str) -> None:
    ks_test_policy_all(policy=policy, policy_name=policy_name)

    snli_data_loader = SNLI_DataLoader_Unclean(f"data/snli_1.0/snli_1.0_test.jsonl")
    mnli_data_loader = SNLI_DataLoader_Unclean(f"data/mnli/multinli_1.0/multinli_1.0_dev_matched.jsonl")
    nli_test_policy(snli_data_loader, "SNLI", policy, policy_name)
    nli_test_policy(mnli_data_loader, "MNLI", policy, policy_name)
    # sick_test_policy_all(policy=policy, policy_name=policy_name)


if __name__ == "__main__":
    # test_policy(comp_policy1(), policy_name="comp_policy")
    policy1 = example_policy()
    policy_name1 = "example_policy"

    test_policy(policy1, policy_name1)


