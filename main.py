import embeddings_library as embed

from dotenv import load_dotenv

from SNLI_data_handling import SNLI_DataLoader
from Hyponyms import KS, Hyponyms, DenseHyponymMatrices

from transformer_library2 import EntailmentNet, HyperParams


def main():
    load_dotenv()

    train_path = "data/snli_1.0/snli_1.0_train.jsonl"
    validation_path = "data/snli_1.0/snli_1.0_dev.jsonl"
    test_path = "data/snli_1.0/snli_1.0_test.jsonl"

    train_loader = SNLI_DataLoader(train_path)

    # This references a GloveEmbedding SQLite Database with 20 - 100 ms word query time.
    # Easily the fastest load time, but not QUITE as fast as compiling for 3 minutes into RAM for < 20 ms query time.
    word_vectors = embed.GloveEmbedding('common_crawl_48', d_emb=300, show_progress=True, default='zero')

    params = HyperParams(heads=10)

    mike_net = EntailmentNet(word_vectors, train_loader, hyper_parameters=params)
    mike_net.train(epochs=100, batch_size=256, print_every=1)


if __name__ == '__main__':
    main()
