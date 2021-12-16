import torch.optim as optim
from dotenv import load_dotenv

from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed
from NLI_hyponomy_analysis.data_pipeline.SNLI_data_handling import SNLI_DataLoaderOptimized
from model_library import HyperParams
from models import NeuralNetwork, StaticEntailmentNet
from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices


def main():
    load_dotenv()

    train_path = "data/snli_1.0/snli_1.0_train.jsonl"
    validation_path = "data/snli_1.0/snli_1.0_dev.jsonl"
    test_path = "data/snli_1.0/snli_1.0_test.jsonl"

    train_small_path = "data/snli_small/snli_small1_train.jsonl"
    validation_small_path = "data/snli_small/snli_small1_dev.jsonl"

    train_loader = SNLI_DataLoaderOptimized(train_small_path)

    validation_loader = SNLI_DataLoaderOptimized(validation_small_path)
    # test_loader = SNLI_DataLoaderOptimized(test_path)

    # word_vectors = embed.GloveEmbedding('twitter', d_emb=25, show_progress=True, default='zero')
    # word_vectors.load_memory()
    # embed.remove_all_non_unique(word_vectors, train_loader.unique_words)

    word_vectors = DenseHyponymMatrices("data/hyponyms/dm-25d-glove-wn.json")
    word_vectors.remove_words(train_loader.unique_words)
    word_vectors.flatten()

    params = HyperParams(heads=5, learning_rate=0.5, dropout=0.3, optimizer=optim.Adadelta,
                         patience=10, early_stopping_mode="minimum")

    mike_net = StaticEntailmentNet(word_vectors, train_loader,
                                   file_path='data/models/nn/test_small_model5.pth',
                                   hyper_parameters=params, classifier_model=NeuralNetwork,
                                   validation_data_loader=validation_loader)

    mike_net.count_parameters()

    # mike_net.train(epochs=100, batch_size=256, print_every=1)
    mike_net.plot_loss()
    mike_net.plot_accuracy()
    # mike_net.test(validation_loader)


if __name__ == '__main__':
    main()
