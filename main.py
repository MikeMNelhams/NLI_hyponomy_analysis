from data_handling import SNLI_DataLoader


def main():
    test_loader = SNLI_DataLoader("data/snli_1.0/snli_1.0_test.jsonl")
    train_loader = SNLI_DataLoader("data/snli_1.0/snli_1.0_train.jsonl")
    validation_loader = SNLI_DataLoader("data/snli_1.0/snli_1.0_dev.jsonl")

    batch_size = 256

    train_data = test_loader.load_batch_random(batch_size)
    print(list(train_data[0].keys()))

    train_sentence1 = train_data.data_by_field("sentence1")
    print(train_sentence1)


if __name__ == '__main__':
    main()



