import unittest
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_Unclean
from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices
from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed

from NLI_hyponomy_analysis.parse_tree import ParseTree

from dotenv import load_dotenv

word_vectors_0 = embed.GloveEmbedding('twitter', d_emb=25, show_progress=True, default='zero')
word_vectors_0.load_memory()


class SNLI_Train_test(unittest.TestCase):
    data_loader = SNLI_DataLoader_Unclean("data/snli_1.0/snli_1.0_train.jsonl")

    load_dotenv()

    word_vectors = DenseHyponymMatrices("data/hyponyms/dm-25d-glove-wn_train_lemma_pos.json")
    word_vectors.remove_all_except(data_loader.unique_words)
    word_vectors.flatten()
    word_vectors.generate_missing_vectors(data_loader.unique_words, word_vectors_0)
    word_vectors.square()

    def test_str_input_none(self):
        batch = self.data_loader.load_line(88117).to_model_data(["sentence1_parse", "sentence2_parse", "gold_label"])

        tree = ParseTree(batch[0][1], word_vectors=self.word_vectors)
        tree.evaluate()
        self.assertEqual(tree.data[0], None)


if __name__ == '__main__':
    unittest.main()
