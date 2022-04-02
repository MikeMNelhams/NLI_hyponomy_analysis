import torch.optim as optim
from dotenv import load_dotenv

from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_POS_Processed
from model_library import HyperParams, Regularisation
from models import NeuralNetwork, StaticEntailmentNet, LSTM, EntailmentTransformer
from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices, Hyponyms


def main():
    load_dotenv()  # Path to the glove data directory -> HOME="..."

    train_path = "data/snli_1.0/snli_1.0_train.jsonl"
    validation_path = "data/snli_1.0/snli_1.0_dev.jsonl"
    test_path = "data/snli_1.0/snli_1.0_test.jsonl"

    train_small_path = "data/snli_small/snli_small1_train.jsonl"
    validation_small_path = "data/snli_small/snli_small1_dev.jsonl"

    train_loader = SNLI_DataLoader_POS_Processed(train_path)
    validation_loader = SNLI_DataLoader_POS_Processed(validation_path)
    test_loader = SNLI_DataLoader_POS_Processed(test_path)

    word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors_0.load_memory()
    # word_vectors_0.remove_all_except(train_loader.unique_words)

    # hyponyms = Hyponyms("data/hyponyms/25d_hyponyms_train_lemma_pos.json", train_loader.unique_words)
    #
    # word_vectors = DenseHyponymMatrices(hyponyms, word_vectors_0.dict)
    # word_vectors.flatten()

    num_layers = 9
    regularisation = Regularisation(l2=0.001)
    params = HyperParams(heads=5, learning_rate=0.001, dropout=0.5, optimizer=optim.Adam,
                         regularisation=regularisation,
                         patience=8, early_stopping_mode="minimum", device='cuda', num_layers=num_layers)

    mike_net = StaticEntailmentNet(word_vectors_0, train_loader,
                                   file_path=f'data/models/lstm/glove_full_model_{num_layers}_layers_random.pth',
                                   hyper_parameters=params, classifier_model=LSTM,
                                   validation_data_loader=validation_loader)
    mike_net.count_parameters()
    mike_net.unlock()
    mike_net.train(epochs=100, batch_size=1024, print_every=1, batch_loading_mode="random")
    mike_net.test(test_loader)


if __name__ == '__main__':
    main()
