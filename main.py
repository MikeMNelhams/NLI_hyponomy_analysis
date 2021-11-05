import embeddings_library as embed

from dotenv import load_dotenv

from SNLI_data_handling import SNLI_DataLoader
from Hyponyms import KS, Hyponyms, DenseHyponymMatrices

from transformer_library2 import EntailmentNet, HyperParams, NeuralNetwork


def main():
    load_dotenv()

    train_path = "data/snli_1.0/snli_1.0_train.jsonl"
    validation_path = "data/snli_1.0/snli_1.0_dev.jsonl"
    test_path = "data/snli_1.0/snli_1.0_test.jsonl"

    train_small_path = "data/snli_small/snli_small1_train.jsonl"

    train_loader = SNLI_DataLoader(train_path)
    # test_loader = SNLI_DataLoader(test_path)

    # The SQLite database querying is pretty much better in every way that matters.
    # Can get a speedup moving the SQLite database onto RAM(-disk) // SSD
    # -------------------------------------------------------------------------------#
    #        SQLite         | 20-100ms QUERY |   2 seconds LOAD | 0   RAM | 5gb disk #
    # Load from file to RAM | 1 - 20ms QUERY | 180 seconds LOAD | 5gb RAM | 5gb disk #
    # -------------------------------------------------------------------------------#
    word_vectors = embed.GloveEmbedding('twitter', d_emb=25, show_progress=True, default='zero')
    params = HyperParams(heads=5, batch_size=64, learning_rate=1, dropout=0.3)

    mike_net = EntailmentNet(word_vectors, train_loader, path='data/models/nn/test_model0.pth',
                             hyper_parameters=params, classifier_model=NeuralNetwork)
    mike_net.count_parameters()
    mike_net.train(epochs=2, print_every=1)
    # mike_net.history.plot_accuracy()
    # mike_net.history.plot_loss()
    # mike_net.test(test_loader)


if __name__ == '__main__':
    main()
