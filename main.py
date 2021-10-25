import embeddings_library as embed

from dotenv import load_dotenv

from SNLI_data_handling import SNLI_DataLoader
from Hyponyms import KS, Hyponyms, DenseHyponymMatrices

from transformer_library import EntailmentTransformer


def main():
    load_dotenv()

    train_path = "data/snli_1.0/snli_1.0_train.jsonl"
    validation_path = "data/snli_1.0/snli_1.0_dev.jsonl"
    test_path = "data/snli_1.0/snli_1.0_test.jsonl"

    train_loader = SNLI_DataLoader(train_path)

    batch_size = 256

    train_data = train_loader.load_batch_random(batch_size).to_model_data()

    train_data.clean_data()
    print('Max sentence lengths:', train_data.max_sentence_lengths)

    # This references a GloveEmbedding SQLite Database with 20 - 100 ms word query time.
    # Easily the fastest load time, but not QUITE as fast as compiling for 3 minutes into RAM for < 20 ms query time.
    word_vectors = embed.GloveEmbedding('common_crawl_48', d_emb=300, show_progress=True, default='zero')

    train, train_masks = train_data.to_tensors(word_vectors)
    print('Train shape:', train.shape)
    print('Train mask shape:', train_masks.shape)

    train_labels = train_data.labels_encoding

    mike_transformer = EntailmentTransformer()


if __name__ == '__main__':
    main()
