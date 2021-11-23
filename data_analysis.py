from NLI_hyponomy_analysis.data_pipeline.SNLI_data_handling import SNLI_DataLoader
from NLI_hyponomy_analysis.data_pipeline.file_operations import file_path_without_extension

from NLI_hyponomy_analysis.data_pipeline.word_operations import remove_punctuation
from NLI_hyponomy_analysis.data_pipeline.word_operations import replace_space_for_underscores, WordParser

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud


class FigureParams:
    def __init__(self, is_standardized=False, number_to_display: int=-1):
        self.is_standardized = is_standardized
        self.number_to_display = number_to_display


class WordFrequencyFigureParams(FigureParams):
    def __init__(self, dataset_name="", metric="Word Frequency", x_label="Words", y_label="Frequency",
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
    def __init__(self, data_loader):
        self.__data_loader = data_loader

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

    def plot_word_histogram(self, sentence_name: str="sentence1", dataset_name: str="Training") -> None:
        figure_params = WordFrequencyFigureParams(dataset_name=f"{dataset_name} data: {sentence_name}")

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

    def plot_boxplot(self, dataset_name: str="Training") -> None:
        figure_params = WordFrequencyFigureParams(dataset_name=dataset_name)

        # Figure labels
        plt.title = f"Boxplot for {dataset_name}"

        # Generate data and plot
        _, frequencies_sentence1 = self.frequency(figure_params, sentence_name="sentence1")
        _, frequencies_sentence2 = self.frequency(figure_params, sentence_name="sentence2")
        frequencies = [frequencies_sentence1, frequencies_sentence2]

        plt.boxplot(frequencies, vert=False)

        # Tweak spacing to prevent clipping of tick-labels
        plt.tight_layout()

        # Saving the plot
        plt.savefig(figure_params.file_save_path + "_boxplot")

        plt.show()
        return None

    def plot_word_cumulative_frequency(self, sentence_name: str="sentence1", dataset_name: str="Training") -> None:
        # TODO Fix this the AXIS ARE FLIPPED!!
        figure_params = WordFrequencyFigureParams(dataset_name=f"{dataset_name} data: {sentence_name}",
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
    train_data_loader = SNLI_DataLoader("data/snli_1.0/snli_1.0_train.jsonl")

    train_data_analysis = DataAnalysis(train_data_loader)

    train_data_analysis.plot_word_histogram("sentence1")
    train_data_analysis.plot_word_histogram("sentence2")


def validation_word_frequency_histogram():
    validation_loader = SNLI_DataLoader("data/snli_1.0/snli_1.0_dev.jsonl")

    validation_loader_analysis = DataAnalysis(validation_loader)

    validation_loader_analysis.plot_word_histogram("sentence1", dataset_name="validation")
    validation_loader_analysis.plot_word_histogram("sentence2", dataset_name="validation")


def test_word_frequency_histogram():
    test_loader = SNLI_DataLoader("data/snli_1.0/snli_1.0_test.jsonl")

    test_loader_analysis = DataAnalysis(test_loader)

    test_loader_analysis.plot_word_histogram("sentence1", dataset_name="test")
    test_loader_analysis.plot_word_histogram("sentence2", dataset_name="test")


def train_word_frequency_boxplot():
    train_loader = SNLI_DataLoader("data/snli_1.0/snli_1.0_train.jsonl")

    train_loader_analysis = DataAnalysis(train_loader)

    train_loader_analysis.plot_boxplot("Training")


def train_word_frequency_cumulative():
    train_data_loader = SNLI_DataLoader("data/snli_1.0/snli_1.0_train.jsonl")

    train_data_analysis = DataAnalysis(train_data_loader)

    train_data_analysis.plot_word_cumulative_frequency("sentence1")


if __name__ == "__main__":
    train_word_frequency_cumulative()
