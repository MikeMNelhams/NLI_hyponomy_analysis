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

    # The SQLite database querying is pretty much better in every way that matters.
    # Can get about speedup moving the SQLite database onto RAM(-disk) // SSD
    # -------------------------------------------------------------------------------#
    #        SQLite         | 20-100ms QUERY |   2 seconds LOAD | 0   RAM | 5gb disk #
    # Load from file to RAM | 1 - 20ms QUERY | 180 seconds LOAD | 5gb RAM | 6gb disk #
    # -------------------------------------------------------------------------------#
    word_vectors = embed.GloveEmbedding('twitter', d_emb=25, show_progress=True, default='zero')
    params = HyperParams(heads=5, batch_size=32)

    mike_net = EntailmentNet(word_vectors, train_loader, hyper_parameters=params)
    mike_net.print_available_devices()
    mike_net.train(epochs=1, print_every=1)


if __name__ == '__main__':
    main()
