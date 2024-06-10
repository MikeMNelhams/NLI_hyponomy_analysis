import unittest
from policies import only_addition, only_addition_no_scaling
from parse_tree import ParseTree
import numpy as np

from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices
import NLI_hyponomy_analysis.data_pipeline.matrix_operations.hyponymy_library as hl


class R2L_Policy(unittest.TestCase):
    def test_addition_only_zeros_returns_none(self):
        policy = only_addition()
        sentence = "(ROOT (S (NP man) (VBZ plays) (NN piano)))"
        word_embeddings = {"man": np.zeros((2, 2)), "plays": np.zeros((2, 2)), "piano": np.zeros((2, 2))}
        word_vectors = DenseHyponymMatrices(None, word_embeddings)
        parse_tree = ParseTree(sentence, word_vectors, policy)
        parse_tree.evaluate()

        self.assertTrue(np.array_equal(parse_tree.data[0], None))

    def test_addition_with_zeros_returns_none(self):
        policy = only_addition()
        sentence = "(ROOT (S (NP man) (VBZ plays) (NN piano)))"
        word_embeddings = {"man": np.identity(2), "plays": np.zeros((2, 2)), "piano": np.zeros((2, 2))}
        word_vectors = DenseHyponymMatrices(None, word_embeddings)
        parse_tree = ParseTree(sentence, word_vectors, policy)
        parse_tree.evaluate()
        self.assertTrue(np.array_equal(parse_tree.data[0], None))

    def test_addition_without_scaling(self):
        policy = only_addition_no_scaling()
        sentence = "(ROOT (S (NP man) (VBZ plays) (NN piano)))"
        word_embeddings = {"man": np.array([[0.2, 0],
                                            [0, 0.2]]),
                           "plays": np.array([[0, 0.1],
                                              [0.3, 0]]),
                           "piano": np.array([[0.4, 0],
                                              [0, 0.5]])}
        word_vectors = DenseHyponymMatrices(None, word_embeddings)
        parse_tree = ParseTree(sentence, word_vectors, policy)
        parse_tree.evaluate()

        self.assertTrue(np.allclose(parse_tree.data[0], np.array([[0.6, 0.1], [0.3, 0.7]])))

    def test_addition_with_scaling(self):
        policy = only_addition()
        sentence = "(ROOT (S (NP man) (VBZ plays) (NN piano)))"
        word_embeddings = {"man": np.array([[0.2, 0],
                                            [0, 0.2]]),
                           "plays": np.array([[0, 0.1],
                                              [0.3, 0]]),
                           "piano": np.array([[0.4, 0],
                                              [0, 0.5]])}
        word_vectors = DenseHyponymMatrices(None, word_embeddings)
        parse_tree = ParseTree(sentence, word_vectors, policy)
        parse_tree.evaluate()

        self.assertFalse(np.allclose(parse_tree.data[0], np.array([[0.6, 0.1], [0.3, 0.7]])))
        self.assertTrue(np.allclose(parse_tree.data[0], np.array([[0.42857143, 0.07142857], [0.21428571, 0.5]])))


class TestingScaling(unittest.TestCase):
    def test_scaling_all_normalised(self):
        policy = only_addition()
        sentence = "(ROOT (S (NP man) (VBZ plays) (NN piano)))"
        word_embeddings = {"man": np.array([[0.2, 0],
                                            [0, 0.2]]),
                           "plays": np.array([[0, 0.1],
                                              [0.3, 0]]),
                           "piano": np.array([[0.4, 0],
                                              [0, 0.5]])}
        word_vectors = DenseHyponymMatrices(None, word_embeddings)
        parse_tree_scale = ParseTree(sentence, word_vectors, policy)

        parse_tree_scale.evaluate()

        self.assertTrue((parse_tree_scale.data[0] <= 1).all() and (parse_tree_scale.data[0] >= 0).all())

    def test_scaling_preserves_k_e(self):
        policy_scale = only_addition()
        policy_no_scale = only_addition_no_scaling()

        sentence1 = "(ROOT (S (NP man) (VBZ plays) (NN piano)))"
        sentence2 = "(ROOT (S (NP man) (VBZ plays) (NN instrument)))"
        word_embeddings = {"man": np.array([[0.2, 0],
                                            [0, 0.2]]),
                            "plays": np.array([[0, 0.1],
                                              [0.3, 0]]),
                            "piano": np.array([[0.4, 0],
                                              [0, 0.5]]),
                            "instrument": np.array([[0.5, 0], [0, 0.1]])}
        word_vectors = DenseHyponymMatrices(None, word_embeddings)
        parse_tree_scale = ParseTree(sentence1, word_vectors, policy_scale)
        parse_tree_no_scale = ParseTree(sentence1, word_vectors, policy_no_scale)
        parse_tree_sentence2 = ParseTree(sentence2, word_vectors, policy_no_scale)

        parse_tree_scale.evaluate()
        parse_tree_no_scale.evaluate()
        parse_tree_sentence2.evaluate()

        k_e_res1 = parse_tree_scale.metric(parse_tree_sentence2, hl.k_e)
        k_e_res2 = parse_tree_no_scale.metric(parse_tree_sentence2, hl.k_e)

        self.assertFalse(np.allclose(parse_tree_scale.data[0], parse_tree_no_scale.data[0]))
        self.assertTrue(np.allclose(k_e_res1, k_e_res2))

    def test_scaling_preserves_k_a(self):
        policy_scale = only_addition()
        policy_no_scale = only_addition_no_scaling()

        sentence1 = "(ROOT (S (NP man) (VBZ plays) (NN piano)))"
        sentence2 = "(ROOT (S (NP man) (VBZ plays) (NN instrument)))"
        word_embeddings = {"man": np.array([[0.2, 0],
                                            [0, 0.2]]),
                           "plays": np.array([[0, 0.1],
                                              [0.3, 0]]),
                           "piano": np.array([[0.4, 0],
                                              [0, 0.5]]),
                           "instrument": np.array([[0.5, 0], [0, 0.1]])}
        word_vectors = DenseHyponymMatrices(None, word_embeddings)
        parse_tree_scale = ParseTree(sentence1, word_vectors, policy_scale)
        parse_tree_no_scale = ParseTree(sentence1, word_vectors, policy_no_scale)
        parse_tree_sentence2 = ParseTree(sentence2, word_vectors, policy_no_scale)

        parse_tree_scale.evaluate()
        parse_tree_no_scale.evaluate()
        parse_tree_sentence2.evaluate()

        k_a_res1 = parse_tree_scale.metric(parse_tree_sentence2, hl.k_ba)
        k_a_res2 = parse_tree_no_scale.metric(parse_tree_sentence2, hl.k_ba)

        self.assertFalse(np.allclose(parse_tree_scale.data[0], parse_tree_no_scale.data[0], atol=0))
        self.assertTrue(np.allclose(k_a_res1, k_a_res2))


if __name__ == '__main__':
    unittest.main()
