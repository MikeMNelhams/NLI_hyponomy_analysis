from SNLI_data_handling import SNLI_DataLoader
from Hyponyms import KS, Hyponyms, DenseHyponymMatrices

from custom_embeddings import glove_matrix


def main():
    train_path = "data/snli_1.0/snli_1.0_train.jsonl"
    validation_path = "data/snli_1.0/snli_1.0_dev.jsonl"
    test_path = "data/snli_1.0/snli_1.0_test.jsonl"

    train_loader = SNLI_DataLoader(train_path)

    batch_size = 256

    train_data = train_loader.load_batch_random(batch_size).to_model_data()

    train_data.clean_data()

    # word_vectors = glove_matrix(input_file_path='data\\embedding_data\\glove\\glove.42B.300d.txt',
    #                             output_file_path='data\\embedding_data\\word2vec\\glove.42B.300d.txt')


if __name__ == '__main__':
    main()
