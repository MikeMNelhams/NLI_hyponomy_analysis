import torch.optim as optim
from dotenv import load_dotenv

from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed
from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_POS_Processed
from model_library import HyperParams, Regularisation
from models import NeuralNetwork, StaticEntailmentNet, LSTM, EntailmentTransformer
from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices, Hyponyms
from sklearn.model_selection import GridSearchCV
import torch

torch.cuda.empty_cache()


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
    word_vectors_0.remove_all_except(train_loader.unique_words)

    hyponyms = Hyponyms("data/hyponyms/25d_hyponyms_train_lemma_pos.json", train_loader.unique_words)

    word_vectors = DenseHyponymMatrices(hyponyms, word_vectors_0.dict)
    word_vectors.flatten()

    num_layers = 6
    l2_coefficient = 0.03

    l2_str = str(l2_coefficient).replace(".", "_")
    regularisation = Regularisation(l2=l2_coefficient)

    model_name = f"data/models/lstm_200/hyp_full_model2_3_layers_random_l2_0_03_BACKUP"

    params = HyperParams(heads=5, learning_rate=0.000_1, dropout=0.4, optimizer=optim.Adam,
                         regularisation=regularisation,
                         patience=10, early_stopping_mode="minimum", device='cuda', num_layers=num_layers)

    mike_net = StaticEntailmentNet(word_vectors, train_loader,
                                   file_path=f'{model_name}.pth',
                                   hyper_parameters=params, classifier_model=NeuralNetwork,
                                   validation_data_loader=validation_loader)
    mike_net.count_parameters()
    mike_net.unlock()
    # mike_net.train(epochs=50, batch_size=512, print_every=1, batch_loading_mode="sequential")
    # mike_net.plot_loss()
    # mike_net.plot_accuracy()
    # mike_net.test(test_loader)
    mike_net.test_confusion_matrix(test_loader, f"{model_name}/confusion_matrix.png", "Confusion Matrix for Simple NN using density matrices")


if __name__ == '__main__':
    main()
