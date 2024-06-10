import unittest
import conditions


class MockLeaf:
    def __init__(self, value, label: str):
        self.value = value
        self._label = label

    def label(self):
        return self._label

    def __getitem__(self, item):
        if item != 0:
            raise IndexError
        return self.value


class UniversalTrue(unittest.TestCase):
    def test_always_true_given_false(self):
        self.assertTrue(conditions.universal_true(False))  # add assertion here

    def test_always_true_given_true(self):
        self.assertTrue(conditions.universal_true(True))

    def test_always_true_given_multiple_args(self):
        self.assertTrue(conditions.universal_true(True, False, False, True))

    def test_always_true_given_none(self):
        self.assertTrue(conditions.universal_true(None))

    def test_always_true_given_no_arg(self):
        self.assertTrue(conditions.universal_true())


class DoesNotContainNoneTree(unittest.TestCase):
    """Uses Mock Trees"""
    def test_returns_true_from_tree_list(self):
        trees = [[1], [2], [3]]
        self.assertTrue(conditions.does_not_contain_none_tree(*trees))

    def test_returns_false_from_tree_list_with_none(self):
        trees = [[1], [None], [3]]
        self.assertFalse(conditions.does_not_contain_none_tree(*trees))

    def test_empty_returns_true(self):
        self.assertTrue(conditions.does_not_contain_none_tree())

    def test_all_none_returns_false(self):
        trees = [[None], [None], [None]]
        self.assertFalse(conditions.does_not_contain_none_tree(*trees))

    def test_single_none_returns_false(self):
        self.assertFalse(conditions.does_not_contain_none_tree(*[[None]]))


class NounPOS(unittest.TestCase):
    """Uses Mock Trees"""
    def test_letter_n_correctly_identified(self):
        test_leaf = MockLeaf(1, "N")
        self.assertTrue(conditions.is_noun(test_leaf))

    def test_noun_correctly_identified(self):
        test_leaf = MockLeaf(1, "NOQWEJIJIWQ")
        self.assertTrue(conditions.is_noun(test_leaf))

    def test_not_noun_correctly_identified_as_false(self):
        test_leaf = MockLeaf(1, "VBP")
        self.assertFalse(conditions.is_noun(test_leaf))

    def test_empty_identified_as_false(self):
        test_leaf = MockLeaf(1, "")
        self.assertFalse(conditions.is_noun(test_leaf))

    def test_word_none_identified_as_false(self):
        test_leaf = MockLeaf(1, "none")
        self.assertFalse(conditions.is_noun(test_leaf))


class VerbPOS(unittest.TestCase):
    """Uses Mock Trees"""
    def test_letter_v_correctly_identified(self):
        test_leaf = MockLeaf(1, "v")
        self.assertTrue(conditions.is_verb(test_leaf))

    def test_verb_correctly_identified(self):
        test_leaf = MockLeaf(1, "VOQWEJIJIWQ")
        self.assertTrue(conditions.is_verb(test_leaf))

    def test_not_verb_correctly_identified_as_false(self):
        test_leaf = MockLeaf(1, "NBP")
        self.assertFalse(conditions.is_verb(test_leaf))

    def test_empty_identified_as_false(self):
        test_leaf = MockLeaf(1, "")
        self.assertFalse(conditions.is_verb(test_leaf))

    def test_word_none_identified_as_false(self):
        test_leaf = MockLeaf(1, "none")
        self.assertFalse(conditions.is_noun(test_leaf))


class NounVerbNoun(unittest.TestCase):
    """ Uses Mock Trees"""
    def test_all_incorrect_phrase_correctly_identified_as_false(self):
        test_leaf1 = MockLeaf(1, "b")
        test_leaf2 = MockLeaf(1, "b")
        test_leaf3 = MockLeaf(1, "b")
        self.assertFalse(conditions.is_noun_verb_noun(test_leaf1, test_leaf2, test_leaf3))

    def test_two_correctly_identified_as_false(self):
        test_leaf1 = MockLeaf(1, "n")
        test_leaf2 = MockLeaf(1, "v")
        test_leaf3 = MockLeaf(1, "b")
        self.assertFalse(conditions.is_noun_verb_noun(test_leaf1, test_leaf2, test_leaf3))

    def test_last_correctly_identified_as_false(self):
        test_leaf1 = MockLeaf(1, "b")
        test_leaf2 = MockLeaf(1, "b")
        test_leaf3 = MockLeaf(1, "n")
        self.assertFalse(conditions.is_noun_verb_noun(test_leaf1, test_leaf2, test_leaf3))

    def test_all_correct_correctly_identified_as_true(self):
        test_leaf1 = MockLeaf(1, "n")
        test_leaf2 = MockLeaf(1, "v")
        test_leaf3 = MockLeaf(1, "n")
        self.assertTrue(conditions.is_noun_verb_noun(test_leaf1, test_leaf2, test_leaf3))

    def test_two_empty_correctly_identified_as_false(self):
        test_leaf1 = MockLeaf(1, "")
        test_leaf2 = MockLeaf(1, "v")
        test_leaf3 = MockLeaf(1, "")
        self.assertFalse(conditions.is_noun_verb_noun(test_leaf1, test_leaf2, test_leaf3))


class Adjective(unittest.TestCase):
    def test_not_adjective(self):
        test_leaf = MockLeaf(1, "v")
        self.assertFalse(conditions.is_adjective(test_leaf))

    def test_empty_phrase_is_not_adjective(self):
        test_leaf = MockLeaf(1, "")
        self.assertFalse(conditions.is_adjective(test_leaf))

    def test_is_adjective(self):
        test_leaf1 = MockLeaf(1, "in")
        test_leaf2 = MockLeaf(1, "jj")
        self.assertTrue(conditions.is_adjective(test_leaf1))
        self.assertTrue(conditions.is_adjective(test_leaf2))


if __name__ == '__main__':
    unittest.main()
