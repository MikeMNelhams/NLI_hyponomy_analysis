import torch.optim as optim
from dotenv import load_dotenv

import matplotlib.pyplot as plt

from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed
from data_pipeline.SNLI_data_handling import SNLI_DataLoaderOptimized
from model_library import HyperParams
from models import NeuralNetwork, StaticEntailmentNet


def main():
    load_dotenv()

    train_path = "data/snli_1.0/snli_1.0_train.jsonl"
    validation_path = "data/snli_1.0/snli_1.0_dev.jsonl"
    test_path = "data/snli_1.0/snli_1.0_test.jsonl"

    train_small_path = "data/snli_small/snli_small1_train.jsonl"
    validation_small_path = "data/snli_small/snli_small1_dev.jsonl"

    train_loader = SNLI_DataLoaderOptimized(train_path)

    validation_loader = SNLI_DataLoaderOptimized(validation_path)
    # test_loader = SNLI_DataLoaderOptimized(test_path)

    # Here is a table of different lookup methods I have tested
    # -----------------------------------------------------------------------------------------------------#
    #    SQLite (Hybrid Disk)               | 120ms - 1000ms QUERY | 002 seconds LOAD | 0   RAM | 6gb disk #
    #    Load from file to RAM              | 100ms - 0200ms QUERY | 180 seconds LOAD | 5gb RAM | 5gb disk #
    #    SQLite to RAM - current solution   | 020ms - 0040ms QUERY | 012 seconds LOAD | 6gb RAM | 6gb disk #
    # -----------------------------------------------------------------------------------------------------#
    word_vectors = embed.GloveEmbedding('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors.load_memory()

    embed.remove_all_non_unique(word_vectors, train_loader.unique_words)

    params = HyperParams(heads=5, learning_rate=0.5, dropout=0.3, optimizer=optim.Adadelta,
                         patience=10, early_stopping_mode="minimum")

    mike_net = StaticEntailmentNet(word_vectors, train_loader,
                                   file_path='data/models/nn/nn_model3.pth',
                                   hyper_parameters=params, classifier_model=NeuralNetwork,
                                   validation_data_loader=validation_loader)

    mike_net.count_parameters()

    mike_net.train(epochs=100, batch_size=1920, print_every=1)

    # mike_net.test(validation_loader)


if __name__ == '__main__':
    main()
