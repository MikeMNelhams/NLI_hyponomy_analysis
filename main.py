from SNLI_data_handling import SNLI_DataLoader
from Hyponyms import KS2016, Hyponyms, DenseHyponymMatrices


def main():
    train_path = "data/snli_1.0/snli_1.0_train.jsonl"
    validation_path = "data/snli_1.0/snli_1.0_dev.jsonl"
    test_path = "data/snli_1.0/snli_1.0_test.jsonl"

    test_loader = SNLI_DataLoader(train_path)

    batch_size = 256

    train_data = test_loader.load_batch_random(batch_size)
    print(train_data.to_sentence_batch(field_name="sentence1"))


if __name__ == '__main__':
    main()
