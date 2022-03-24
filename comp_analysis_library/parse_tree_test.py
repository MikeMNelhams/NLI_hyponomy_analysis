import unittest
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_Unclean
from NLI_hyponomy_analysis.data_pipeline.hyponyms import Hyponyms, DenseHyponymMatrices
from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed

from NLI_hyponomy_analysis.compositional_analysis.parse_tree import ParseTree

from dotenv import load_dotenv

load_dotenv()
train_loader = SNLI_DataLoader_Unclean("../data/snli_1.0/snli_1.0_train.jsonl")

word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
word_vectors_0.load_memory()


class SNLI_Train_test(unittest.TestCase):
    word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()
    word_vectors_0.remove_all_except(train_loader.unique_words)

    hyponyms = Hyponyms("../data/hyponyms/25d_hyponyms_train_lemma_pos.json", train_loader.unique_words)

    word_vectors = DenseHyponymMatrices(hyponyms, word_vectors_0.dict)

    def test_str_input_none(self):
        batch = train_loader.load_line(88117).to_model_data(["sentence1_parse", "sentence2_parse", "gold_label"])

        tree = ParseTree(batch[0][1], word_vectors=self.word_vectors)
        tree.evaluate()
        self.assertEqual(tree.data[0], None)


if __name__ == '__main__':
    unittest.main()
