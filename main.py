from data_handling import SNLI_DataLoader
import word_embedding


def main():
    test_loader = SNLI_DataLoader("data/snli_1.0/snli_1.0_test.jsonl")

    batch_size = 256

    train_data = test_loader.load_batch_random(batch_size)
    print(list(train_data[0].keys()))

    train_data.data_by_field("sentence1")
    print(train_data)
    train_data.apply(word_embedding.sentence_to_words)
    print(train_data)


if __name__ == '__main__':
    main()
