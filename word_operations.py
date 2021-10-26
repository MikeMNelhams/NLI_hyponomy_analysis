from typing import Iterable
from typing import List


def remove_punctuation(word: str, punctuation: iter = (',', '.')) -> str:
    returned_word = word
    for symbol in punctuation:
        returned_word = returned_word.replace(symbol, '')
    return returned_word


def replace_ampersand(word: str) -> str:
    return word.replace('&', 'and')


def remove_speech_marks(word: str) -> str:
    return word.replace('"', '')


def count_max_sequence_length(list_of_sentences: List[str]):
    return max(len(row.split()) for row in list_of_sentences)


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
        return WordParser((str.lower, replace_ampersand, remove_punctuation, remove_speech_marks, str.strip))
