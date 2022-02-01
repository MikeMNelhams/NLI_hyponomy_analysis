from typing import Iterable
from typing import List

import re
from nltk.stem import WordNetLemmatizer


def replace_space_for_underscores(word: str) -> str:
    return word.replace(' ', '_')


def remove_punctuation(word: str, punctuation: iter = (',', '.', '\'', '\"', "*", "?", "!", ":")) -> str:
    returned_word = word
    for symbol in punctuation:
        returned_word = returned_word.replace(symbol, '')
    return returned_word


def regex_clean_all_punctuation(word: str) -> str:

    returned_word = re.sub(r"[^A-Za-z]+", ' ', word)

    return returned_word


def find_all_pos_tags(sentence: str) -> List[str]:
    tags = re.findall(r"\([A-Za-z]* [A-Za-z]*\)", sentence)
    tags = [tag[1:-1] for tag in tags]
    return tags


def replace_ampersand(word: str) -> str:
    return word.replace('&', 'and')


def remove_speech_marks(word: str) -> str:
    return word.replace('"', '')


def count_max_sequence_length(list_of_sentences: List[str]):
    return max(len(row.split()) for row in list_of_sentences)


def lemmatise_sentence(sentence: str) -> str:
    lemmatiser = WordNetLemmatizer()
    words = sentence.split(' ')
    words = [lemmatiser.lemmatize(word) for word in words]
    return ' '.join(words)


def lemmatise_sentence_pos(sentence: str) -> str:
    lemmatiser = WordNetLemmatizer()
    pos = find_all_pos_tags(sentence)

    def split_tags(tag):
        for i in range(len(tag)):
            if tag[i] == ' ':
                return [tag[:i], tag[i + 1:]]
        raise TypeError

    def lemmatise(word: str, tag: str) -> str:
        first_letter = tag[0]
        valid_tags = ('a', 'n', 'v', 'r', 's')

        if first_letter in valid_tags:
            return lemmatiser.lemmatize(word, pos=first_letter)

        return lemmatiser.lemmatize(word)

    pos = [split_tags(tag.lower()) for tag in pos]

    words = [lemmatise(part[1], part[0]) for part in pos]

    return ' '.join(words)


class WordParser:
    """ A way to combine multiple filters into a callable."""
    def __init__(self, actions: Iterable[callable]):
        self.__actions = actions

    def __call__(self, word: str) -> str:
        word_copy = word
        for action in self.__actions:
            word_copy = action(word_copy)
        return word_copy

    @staticmethod
    def default_clean() -> "WordParser":
        return WordParser((regex_clean_all_punctuation, str.lower, str.strip))

    @staticmethod
    def default_lemmatisation() -> "WordParser":
        return WordParser([lemmatise_sentence])

    @staticmethod
    def default_lemmatisation_pos() -> "WordParser":
        return WordParser([lemmatise_sentence_pos])
