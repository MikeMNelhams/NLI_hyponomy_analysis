from NLI_hyponomy_analysis.data_pipeline.SNLI_data_handling import SNLI_DataLoader
from NLI_hyponomy_analysis.data_pipeline.file_operations import file_path_without_extension

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud


def create_wordcloud(text: str, mask_file_path: str, save_name: str = "0") -> None:
    mask = np.array(Image.open(mask_file_path))

    wc = WordCloud(background_color="white",
                   mask=mask,
                   max_words=200)

    wc.generate(text)

    containing_folder = file_path_without_extension(mask_file_path)

    wc.to_file(containing_folder + '_' + save_name)
    return None


def plot_histogram(term_count: dict, number_of_frequencies_to_display: int = 25, standardized=True) -> None:
    frequencies = np.array(list(term_count.items()))
    # Sort by frequency, reversed
    frequencies = frequencies[frequencies[:, 1].astype(float).argsort()[::-1]]

    # Only take the top N highest frequencies
    values = frequencies[:number_of_frequencies_to_display, 1].astype(float)

    if standardized:
        values = (values - np.mean(values)) / float(np.std(values))

    x_ticks = frequencies[: number_of_frequencies_to_display, 0]

    plt.bar(x_ticks, values)
    plt.xticks(x_ticks, x_ticks, rotation='vertical')

    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.1)

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)

    plt.show()
    return None


def main():
    train_data_loader = SNLI_DataLoader("data/snli_1.0/snli_1.0_train.jsonl")

    train_s1_term_count = train_data_loader.term_count("sentence1")
    # train_s2_term_count = train_data_loader.term_count("sentence2")

    plot_histogram(train_s1_term_count, standardized=False)


if __name__ == "__main__":
    main()
