import os.path

from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_POS_Processed
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_Unclean, SNLI_DataLoader_Processed
from NLI_hyponomy_analysis.data_pipeline.file_operations import file_path_without_extension

from NLI_hyponomy_analysis.data_pipeline.word_operations import remove_punctuation
from NLI_hyponomy_analysis.data_pipeline.word_operations import replace_space_for_underscores, WordParser

from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud


class FigureParams:
    def __init__(self, is_standardized=False, number_to_display: int=-1):
        self.is_standardized = is_standardized
        self.number_to_display = number_to_display


class FrequencyFigureParams(FigureParams):
    def __init__(self, dataset_name="", metric="Frequency", x_label="Words", y_label="Frequency",
                 is_standardized=False, number_to_display: int = 25):

        super().__init__(is_standardized)

        # Labels
        self.title = f"{metric} for the {dataset_name}"
        self.x_label = x_label
        self.y_label = y_label

        # Plot parameters
        self.number_to_display = number_to_display

        # Saving the figure
        clean = WordParser([str.lower, remove_punctuation, str.strip, replace_space_for_underscores])
        save_name = clean(dataset_name)
        self.file_save_path = "data/snli_1.0/word_frequency_figures/" + save_name + "_word_freq"


class DataAnalysis:
    def __init__(self, file_path: str):
        assert os.path.isfile(file_path)
        self.__data_loader = SNLI_DataLoader_POS_Processed(file_path)

    @property
    def data_loader(self):
        return self.__data_loader

    def term_count(self, sentence_name: str="sentence1"):
        term_count: dict = self.__data_loader.term_count(sentence_name)
        return term_count

    def frequency(self, figure_params: FigureParams, sentence_name: str="sentence1") -> [np.array, np.array]:
        term_count = self.term_count(sentence_name)

        frequencies = np.array(list(term_count.items()))

        # Sort by frequency, reversed
        frequencies = frequencies[frequencies[:, 1].astype(float).argsort()[::-1]]

        # Only take the top N highest frequencies
        words = frequencies[: figure_params.number_to_display, 0]
        frequencies = frequencies[:figure_params.number_to_display, 1].astype(float)

        if figure_params.is_standardized:
            frequencies = (frequencies - np.mean(frequencies)) / float(np.std(frequencies))

        return words, frequencies

    def label_frequency(self, figure_params: FigureParams) -> [np.array, np.array]:
        label_count = self.__data_loader.label_count()

        labels = np.array(list(label_count.keys()))
        frequencies = np.array(list(label_count.values()))

        if figure_params.is_standardized:
            frequencies = frequencies / float(np.std(frequencies))

        return labels, frequencies

    def plot_word_histogram(self, sentence_name: str="sentence1", dataset_name: str="Training") -> None:
        figure_params = FrequencyFigureParams(dataset_name=f"{dataset_name} data: {sentence_name}")

        x_ticks, frequencies = self.frequency(figure_params, sentence_name=sentence_name)

        plt.bar(x_ticks, frequencies)
        plt.xticks(x_ticks, x_ticks, rotation='vertical')

        # Figure labels
        plt.title(figure_params.title)
        plt.xlabel(figure_params.x_label)
        plt.ylabel(figure_params.y_label)

        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.1)

        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()

        # Saving the plot
        plt.savefig(figure_params.file_save_path)

        plt.show()
        return None

    def plot_word_boxplot(self, dataset_name: str="Training") -> None:
        figure_params = FrequencyFigureParams(dataset_name=dataset_name,
                                              x_label="Word Frequency",
                                              y_label="Sentence Number")

        # Generate data and plot
        _, frequencies_sentence1 = self.frequency(figure_params, sentence_name="sentence1")
        _, frequencies_sentence2 = self.frequency(figure_params, sentence_name="sentence2")
        frequencies = [frequencies_sentence1, frequencies_sentence2]

        plt.boxplot(frequencies, vert=False, showfliers=False)

        # Figure labels
        plt.title(f"Boxplot for {dataset_name}")
        plt.xlabel(figure_params.x_label)
        plt.ylabel(figure_params.y_label)

        # Tweak spacing to prevent clipping of tick-labels
        plt.tight_layout()

        # Saving the plot
        plt.savefig(figure_params.file_save_path + "_boxplot")

        plt.show()
        return None

    def plot_word_cumulative_frequency(self, sentence_name: str="sentence1", dataset_name: str="Training") -> None:
        # TODO Fix this the AXES ARE FLIPPED!!
        figure_params = FrequencyFigureParams(dataset_name=f"{dataset_name} data: {sentence_name}",
                                              number_to_display=-1, is_standardized=True)

        _, frequencies = self.frequency(figure_params, sentence_name=sentence_name)
        frequencies = frequencies.T
        bins = np.append(np.linspace(frequencies.min(), frequencies.max(), len(frequencies) // 2), [np.inf])

        plt.hist(frequencies, bins=bins, histtype="step", cumulative=True)

        # Figure labels
        plt.title(figure_params.title)
        plt.xlabel("Normalized Eigenvalues")
        plt.ylabel("Relative Cumulative Frequency")

        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()

        # Saving the plot
        plt.savefig(figure_params.file_save_path + "_cumulative")

        plt.show()
        return None

    def plot_label_histogram(self, dataset_name: str="Training", color="blue") -> None:
        figure_params = FrequencyFigureParams(dataset_name=f"{dataset_name} labels", is_standardized=True,
                                              x_label="Label", y_label="Relative Frequency")

        x_ticks, frequencies = self.label_frequency(figure_params)

        x_ticks[0] = 'unknown'

        plt.bar(x_ticks, frequencies, color=color)
        plt.xticks(x_ticks, x_ticks, rotation='vertical')

        # Figure labels
        plt.title(figure_params.title)
        plt.xlabel(figure_params.x_label)
        plt.ylabel(figure_params.y_label)

        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.1)

        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)
        plt.tight_layout()

        # Saving the plot
        plt.savefig(figure_params.file_save_path)

        plt.show()
        return None


def create_wordcloud(text: str, mask_file_path: str, save_name: str = "0") -> None:
    mask = np.array(Image.open(mask_file_path))

    wc = WordCloud(background_color="white",
                   mask=mask,
                   max_words=200)

    wc.generate(text)

    containing_folder = file_path_without_extension(mask_file_path)

    wc.to_file(containing_folder + '_' + save_name)
    return None


def training_word_frequency_histogram():
    train_data_analysis = DataAnalysis("data/snli_1.0/snli_1.0_train.jsonl")

    train_data_analysis.plot_word_histogram("sentence1")
    train_data_analysis.plot_word_histogram("sentence2")


def train_word_frequency_boxplot():
    train_loader_analysis = DataAnalysis("data/snli_1.0/snli_1.0_train.jsonl")

    train_loader_analysis.plot_word_boxplot("Training")


def train_word_frequency_cumulative():
    train_data_analysis = DataAnalysis("data/snli_1.0/snli_1.0_train.jsonl")

    train_data_analysis.plot_word_cumulative_frequency("sentence1")


def train_label_histogram():
    training_data_analysis = DataAnalysis("data/snli_1.0/snli_1.0_train.jsonl")

    training_data_analysis.plot_label_histogram("Training", color="blue")


def validation_label_histogram():
    validation_data_analysis = DataAnalysis("data/snli_1.0/snli_1.0_dev.jsonl")

    validation_data_analysis.plot_label_histogram("Validation", color="red")


def validation_word_frequency_histogram():
    validation_loader_analysis = DataAnalysis("data/snli_1.0/snli_1.0_dev.jsonl")

    validation_loader_analysis.plot_word_histogram("sentence1", dataset_name="validation")
    validation_loader_analysis.plot_word_histogram("sentence2", dataset_name="validation")


def validation_word_frequency_boxplot():
    train_loader_analysis = DataAnalysis("data/snli_1.0/snli_1.0_dev.jsonl")

    train_loader_analysis.plot_word_boxplot("Validation")


def test_word_frequency_histogram():
    test_loader_analysis = DataAnalysis("data/snli_1.0/snli_1.0_test.jsonl")

    test_loader_analysis.plot_word_histogram("sentence1", dataset_name="test")
    test_loader_analysis.plot_word_histogram("sentence2", dataset_name="test")


def percentage_hypernym():
    train_path = "data/snli_1.0/snli_1.0_train.jsonl"
    validation_path = "data/snli_1.0/snli_1.0_dev.jsonl"
    test_path = "data/snli_1.0/snli_1.0_test.jsonl"

    data_loader = SNLI_DataLoader_Unclean(train_path)

    word_vectors = DenseHyponymMatrices("data/hyponyms/dm-25d-glove-wn_train_unclean.json")

    train_percent_hypernym = word_vectors.valid_hypernym_percentage(data_loader.unique_words)
    print(train_percent_hypernym)


if __name__ == "__main__":
    percentage_hypernym()
