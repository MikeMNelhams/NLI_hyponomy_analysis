from typing import Iterable


def remove_punctuation(word: str, punctuation: iter = (',', '.')) -> str:
    returned_word = word
    for symbol in punctuation:
        returned_word = returned_word.replace(symbol, '')
    return returned_word


def replace_ampersand(word: str) -> str:
    return word.replace('&', 'and')


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
        return WordParser((str.lower, replace_ampersand, remove_punctuation, str.strip))
