import embeddings_library as embed

from dotenv import load_dotenv

from data_pipeline.SNLI_data_handling import SNLI_DataLoader

from transformer_library2 import EntailmentNet, HyperParams, NeuralNetwork

import cProfile


def main():
    load_dotenv()

    train_path = "data/snli_1.0/snli_1.0_train.jsonl"
    validation_path = "data/snli_1.0/snli_1.0_dev.jsonl"
    test_path = "data/snli_1.0/snli_1.0_test.jsonl"

    train_small_path = "data/snli_small/snli_small1_train.jsonl"

    train_loader = SNLI_DataLoader(train_small_path)
    # test_loader = SNLI_DataLoader(test_path)

    # The SQLite database querying is slower to query, faster to load
    # Can get major speedup moving the SQLite database onto SSD
    # If you have spare RAM, put the SQLite database onto RAM ONCE, then when you log-off, remember to unfreeze from RAM
    # -------------------------------------------------------------------------------#
    #        SQLite (Hybrid Disk)   | 120ms - 1000 QUERY |   2 seconds LOAD | 0   RAM | 6gb disk #
    # Load from file to RAM         | 1 - 20ms QUERY     | 180 seconds LOAD | 5gb RAM | 5gb disk #
    # 
    # -------------------------------------------------------------------------------#
    word_vectors = embed.GloveEmbedding('twitter', d_emb=25, show_progress=True, default='zero')
    # word_vectors = embed.glove_matrix('data/embedding_data/glove/glove.6B.50d.txt',
    #                                   'data/embedding_data/word2vec/glove.6B.50d.txt')
    params = HyperParams(heads=5, batch_size=64, learning_rate=1, dropout=0.3)

    mike_net = EntailmentNet(word_vectors, train_loader, path='data/models/nn/test_small_model_gpu.pth',
                             hyper_parameters=params, classifier_model=NeuralNetwork)
    mike_net.count_parameters()
    mike_net.train(epochs=100, print_every=1)

    # mike_net.history.plot_accuracy()
    # mike_net.history.plot_loss()
    # mike_net.test(test_loader)


if __name__ == '__main__':
    cProfile.run('main()')
