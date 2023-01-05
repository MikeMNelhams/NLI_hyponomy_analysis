from typing import Iterable
from typing import List

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import treebank
from nltk.tag import UnigramTagger
import itertools

# Constants and models
tree_tagger = UnigramTagger(treebank.tagged_sents()[:2500])
label_mapping = {"f": "contradiction", "t": "entailment", "-": "unknown",
                 "contradiction": "contradiction", "entailment": "entailment", "neutral": "neutral"}
bad_chars = ['\u00a0']
pos_tags = {"ROOT", "(", ")", ".", "CC", "CD", "DT", "EX", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN",
            "NNP", "NNS", "NP", "PP", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "S", "TO",
            "UH", "VB", "VP", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WRB", "FRAG", "ADJP", "ADVP",
            "SBAR"}


def standardise_label(label: str) -> str:
    return label_mapping[label]


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


def split_by_pos_tags(sentence: str) -> (List[str], List[str]):
    tags = re.findall(r"\([A-Za-z]* [A-Za-z]*\)", sentence)
    tags = [tag[1:-1] for tag in tags]
    tag_string = r'|'.join(tags)
    remaining = re.split(tag_string, sentence)
    return remaining, tags


def interleave(*lists) -> list:
    return [x for x in itertools.chain(*itertools.zip_longest(*lists)) if x is not None]


def pos_string_to_literal(pos_string) -> str:
    string_literal = ''
    for i in range(len(pos_string) - 1):
        char0 = pos_string[i]
        char1 = pos_string[i + 1]
        if char0 == '(':
            string_literal += "(\""
        elif char0 == ' ' and char1 != '(' and char1 != ')':
            string_literal += " \""
        elif char1 == ' ' and char0 != '(' and char0 != ')':
            string_literal += f"{char0}\""
        elif char1 == ')' and char0 != '(' and char0 != ')':
            string_literal += f"{char0}\""
        else:
            string_literal += char0

    return string_literal


def pos_string_to_tokens_list(pos_string) -> list:
    """ Time O(n). Space O(n), where n is len(string)"""
    token_list = []

    i = 0
    while i < len(pos_string):
        char = pos_string[i]
        if char == ' ':
            pass
        elif not __is_pos_sep(char):
            end_idx = i + 1

            while True:
                end_char = pos_string[end_idx]
                if __is_pos_sep(end_char):
                    break
                elif end_char == ' ':
                    break
                end_idx += 1

            word = pos_string[i: end_idx]
            token_list.append(word)

            i = end_idx
        else:
            token_list.append(char)
        i += 1

    return token_list


def pos_string_to_tokens_list_with_space(pos_string) -> list:
    """ Time O(n). Space O(n), where n is len(string)"""
    token_list = []

    def case1(token_p, token_n):
        return token_p != ' ' and token_n != ')'

    def case2(token_p, token_n):
        return not token_p == '(' or __is_non_alphanumeric_token(token_n)

    def case3(token_p, token_n):
        return not (token_p == '(' and token_n == '(')

    i = 0

    while i < len(pos_string):
        char = pos_string[i]
        if not __is_non_alphanumeric_token(char):
            end_idx = i + 1

            while True:
                end_char = pos_string[end_idx]
                if __is_pos_sep(end_char):
                    break
                elif end_char == ' ':
                    break
                end_idx += 1

            word = pos_string[i: end_idx]
            token_list.append(word)

            i = end_idx - 1
        else:
            token_list.append(char)
        i += 1

    # Need to remove the double spacings and spaces that are next to the brackets.
    corrected_token_list = [token_list[0]]

    for i in range(1, len(token_list) - 1):
        current_token = token_list[i]
        if current_token == ' ':
            previous_token = token_list[i-1]
            next_token = token_list[i+1]
            if case1(previous_token, next_token) and \
                    case2(previous_token, next_token) and case3(previous_token, next_token):
                corrected_token_list.append(current_token)
        else:
            corrected_token_list.append(current_token)

    corrected_token_list.append(token_list[-1])

    return corrected_token_list


def __is_pos_sep(char) -> bool:
    return char == '(' or char == ')'


def __is_non_alphanumeric_token(char) -> bool:
    return __is_pos_sep(char) or char == ' '


def replace_ampersand(word: str) -> str:
    return word.replace('&', 'and')


def remove_speech_marks(word: str) -> str:
    return word.replace('"', '')


def replace_commas_for_periods(word: str) -> str:
    return word.replace(',', '.')


def remove_utf8_bad_chars(word: str) -> str:
    return remove_punctuation(word, bad_chars)


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


def lemmatise_sentence_pos_tag(sentence: str) -> str:
    lemmatiser = WordNetLemmatizer()
    remaining, tags = split_by_pos_tags(sentence)

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

    pos = [split_tags(tag.lower()) for tag in tags]
    words = [f"{part[0].upper()} {lemmatise(part[1], part[0])}" for part in pos]
    lemmatised_sentence = ''.join(interleave(remaining, words))
    return lemmatised_sentence


def pos_tag_sentence(sentence: str) -> str:
    """Creates a flat sentence POS tagged"""
    cleaned_sentence = WordParser.default_clean()(sentence)
    tagged_sentence = tree_tagger.tag(cleaned_sentence.split(' '))
    tagged_sentence = [f"({string_pair[1]} {string_pair[0]})" for string_pair in tagged_sentence]
    tagged_sentence = '(ROOT ' + ' '.join(tagged_sentence) + ' (. .))'
    return tagged_sentence


class InvalidProcessingMode(Exception):
    """ When a given processing mode is not implemented yet"""
    pass


class ProcessingSynonyms:
    synonyms_for_l = ("lemmatise", "lemmatised", "lemma")
    synonyms_for_cl = ("clean_lemmatise", "clean_lemmatised", "cl", "both", "clean_lemma")
    synonyms_for_l_pos = ("lemma_pos", "lemmatised_pos", "l_pos", "lpl", "lemmatise_pos")
    synonyms_for_l_pos_tag = ("lemma_pos_tag", "lemmatised_pos_tag", "l_pos_tag", "lplt", "lemmatise_pos_tag")

    @staticmethod
    def map_processing_mode(mode: str) -> str:
        if mode in ProcessingSynonyms.synonyms_for_l:
            return "lemmatised"
        if mode in ProcessingSynonyms.synonyms_for_cl:
            return "clean_lemmatised"
        if mode in ProcessingSynonyms.synonyms_for_l_pos:
            return "lemmatised_pos"

        raise InvalidProcessingMode


class WordParser:
    """ A way to combine multiple filters into a callable."""
    def __init__(self, actions: Iterable[callable]):
        self.__actions = list(actions)

    def __call__(self, word: str) -> str:
        word_copy = word
        for action in self.__actions:
            word_copy = action(word_copy)
        return word_copy

    def append(self, action: callable):
        self.__actions.append(action)

    @staticmethod
    def default_clean() -> "WordParser":
        return WordParser((regex_clean_all_punctuation, str.lower, str.strip))

    @staticmethod
    def default_lemmatisation() -> "WordParser":
        return WordParser([lemmatise_sentence])

    @staticmethod
    def default_lemmatisation_pos() -> "WordParser":
        return WordParser([lemmatise_sentence_pos])

    @staticmethod
    def default_lemmatisation_pos_tag() -> "WordParser":
        return WordParser([lemmatise_sentence_pos_tag, replace_commas_for_periods])

    @staticmethod
    def default_remove_bar_chars() -> "WordParser":
        return WordParser([remove_utf8_bad_chars])
