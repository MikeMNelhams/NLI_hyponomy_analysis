import shutil
import unittest

import os.path
from dotenv import load_dotenv

from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_POS_Processed, SNLI_DataLoader_Unclean
from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed
from NLI_hyponomy_analysis.data_pipeline.hyponyms import DenseHyponymMatrices2, Hyponyms
import NLI_hyponomy_analysis.data_pipeline.file_operations as file_op

import model_library as ml
from models import StaticEntailmentNet, NeuralNetwork, EntailmentTransformer, HyperParams
import model_errors
import torch.optim as optim
import numpy as np


class TestTeardown:
    def __init__(self, dir_path: str):
        self.__dir_path = dir_path

    def __delete_all(self) -> None:
        if os.path.isdir(self.__dir_path):
            shutil.rmtree(self.__dir_path)
        return None

    def __enter__(self):
        self.__delete_all()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__delete_all()


class NeuralNet(unittest.TestCase):
    load_dotenv()

    train_small_path = "data/snli_small/snli_small1_train.jsonl"
    validation_small_path = "data/snli_small/snli_small1_dev.jsonl"

    train_loader = SNLI_DataLoader_POS_Processed(train_small_path)
    validation_loader = SNLI_DataLoader_POS_Processed(validation_small_path)

    word_vectors = embed.GloveEmbedding('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors.load_memory()

    def test_train(self):
        train_save_path = 'data/test_data/test_train'

        with TestTeardown(train_save_path):
            params = ml.HyperParams(heads=5, learning_rate=0.1, dropout=0.3, optimizer=optim.Adam)
            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params, classifier_model=NeuralNetwork)
            mike_net.train(epochs=100, print_every=10)

            self.assertGreater(mike_net.history.accuracy[-1], 0.3)
            self.assertLess(mike_net.history.loss[-1], 3)
            self.assertLess(mike_net.info.runtime, 10)  # Red flag if it takes longer than 10 seconds.

    def test_loading(self):
        mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path='data/test_data/test_load.pth',
                                       classifier_model=NeuralNetwork)
        print('HISTORY:', mike_net.history.accuracy)

        self.assertGreater(mike_net.history.accuracy[-1], 0.3)
        self.assertLess(mike_net.history.loss[-1], 3)
        self.assertLess(mike_net.info.runtime, 10)  # Red flag if it takes longer than 10 seconds.

    def test_testing(self):
        train_small_path = "data/snli_small/snli_small1_train.jsonl"
        validation_small_path = "data/snli_small/snli_small1_dev.jsonl"

        train_loader = SNLI_DataLoader_POS_Processed(train_small_path)
        validation_loader = SNLI_DataLoader_POS_Processed(validation_small_path)

        word_vectors_0 = embed.Embedding2('twitter', d_emb=25, show_progress=True, default='zero')
        word_vectors_0.load_memory()
        embed.remove_all_except(word_vectors_0, train_loader.unique_words)

        hyponyms = Hyponyms("data/hyponyms/25d_hyponyms_all.json", train_loader.unique_words)

        word_vectors = DenseHyponymMatrices2(hyponyms, word_vectors_0.dict)
        word_vectors.remove_all_except(train_loader.unique_words)
        word_vectors.flatten()
        word_vectors.generate_missing_vectors(train_loader.unique_words, word_vectors_0)

        params = HyperParams(heads=5, learning_rate=0.5, dropout=0.3, optimizer=optim.Adadelta,
                             patience=6, early_stopping_mode="minimum", device='cpu')

        mike_net = StaticEntailmentNet(word_vectors, train_loader,
                                       file_path='data/test_data/test_load.pth',
                                       classifier_model=NeuralNetwork, hyper_parameters=params,
                                       validation_data_loader=validation_loader)

        loss, acc = mike_net.test(self.train_loader)
        print(f'LOSS: {loss}, ACC: {acc}')
        self.assertGreater(mike_net.history.accuracy[-1], 0.3)
        self.assertLess(mike_net.history.loss[-1], 1)

    def test_train_batch_number_too_high(self):
        train_save_path = 'data/test_data/test_train1'

        with TestTeardown(train_save_path):
            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           classifier_model=NeuralNetwork)
            mike_net.train(1)

    def test_train_validation(self):
        train_save_path = 'data/test_data/test_train2'

        with TestTeardown(train_save_path):
            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           classifier_model=NeuralNetwork, validation_data_loader=self.train_loader)
            mike_net.train(1)

    def test_print_available_devices(self):
        train_save_path = 'data/test_data/test_train2'

        mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                       classifier_model=NeuralNetwork, validation_data_loader=self.train_loader)
        mike_net.print_available_devices()

    def test_early_stopping_minimum(self):
        train_save_path = 'data/test_data/test_train'

        with TestTeardown(train_save_path):
            params = HyperParams(heads=5, learning_rate=0.5, dropout=0.3, optimizer=optim.Adadelta,
                                 patience=10, early_stopping_mode="minimum")

            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           classifier_model=NeuralNetwork, validation_data_loader=self.train_loader,
                                           hyper_parameters=params)
            mike_net.train(500, batch_size=256)
            # Early stopping means must not overfit on validation data.
            # The number of epochs should be less than 500.
            self.assertLess(len(mike_net.validation_history.accuracy), 500)
            self.assertGreater(mike_net.validation_history.accuracy[-1], 0.3)

    def test_early_stopping_strict(self):
        train_save_path = 'data/test_data/test_train'

        with TestTeardown(train_save_path):
            params = HyperParams(heads=5, learning_rate=0.5, dropout=0.3, optimizer=optim.Adadelta,
                                 patience=10, early_stopping_mode="minimum")

            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           classifier_model=NeuralNetwork, validation_data_loader=self.train_loader,
                                           hyper_parameters=params)
            mike_net.train(500, batch_size=256)
            # Early stopping means must not overfit on validation data.
            # Strict early stopping should stop less than ~300
            self.assertLess(len(mike_net.validation_history.accuracy), 300)
            self.assertGreater(mike_net.validation_history.accuracy[-1], 0.3)


class Transformer(unittest.TestCase):
    train_small_path = "data/snli_small/snli_small1_train.jsonl"
    train_loader = SNLI_DataLoader_Unclean(train_small_path)
    validation_loader = SNLI_DataLoader_Unclean(train_small_path)

    load_dotenv()

    word_vectors = embed.GloveEmbedding('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors.load_memory()

    def test_train(self):
        train_save_path = 'data/test_data/test_train3'

        with TestTeardown(train_save_path):
            params = ml.HyperParams(heads=5, learning_rate=1, dropout=0.3, optimizer=optim.Adadelta)
            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params, classifier_model=EntailmentTransformer)
            mike_net.train(epochs=100, print_every=10)

            self.assertGreater(mike_net.history.accuracy[-1], 0.3)
            self.assertLess(mike_net.history.loss[-1], 3)
            self.assertLess(mike_net.info.runtime, 10)  # Red flag if it takes longer than 10 seconds.

    def test_loading(self):
        mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader,
                                       file_path='data/test_data/test_load.pth',
                                       classifier_model=EntailmentTransformer)

        self.assertGreater(mike_net.history.accuracy[-1], 0.3)
        self.assertLess(mike_net.history.loss[-1], 3)
        self.assertLess(mike_net.info.runtime, 10)  # Red flag if it takes longer than 10 seconds.

    def test_testing(self):
        mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path='data/test_data/test_load_transformer.pth',
                                       classifier_model=EntailmentTransformer, validation_data_loader=self.train_loader)

        loss, acc = mike_net.test(self.train_loader)
        print(f'LOSS: {loss}, ACC: {acc}')
        self.assertGreater(mike_net.history.accuracy[-1], 0.5)
        self.assertLess(mike_net.history.loss[-1], 1)

    def test_train_batch_number_too_high(self):
        train_save_path = 'data/test_data/test_train4'

        with TestTeardown(train_save_path):
            params = ml.HyperParams(heads=5, learning_rate=1, dropout=0.3, optimizer=optim.Adadelta)
            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params,
                                           classifier_model=EntailmentTransformer)
            mike_net.train(1)

    def test_train_validation(self):
        train_save_path = 'data/test_data/test_train5'

        with TestTeardown(train_save_path):
            params = HyperParams(heads=5, learning_rate=0.5, dropout=0.3, optimizer=optim.Adadelta,
                                 patience=3)
            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params,
                                           classifier_model=EntailmentTransformer,
                                           validation_data_loader=self.train_loader)
            mike_net.train(1)


class BasicProperties(unittest.TestCase):
    train_small_path = "data/snli_small/snli_small1_train.jsonl"
    train_loader = SNLI_DataLoader_Unclean(train_small_path)
    validation_loader = SNLI_DataLoader_Unclean(train_small_path)

    load_dotenv()

    word_vectors = embed.GloveEmbedding('twitter', d_emb=25, show_progress=True, default='zero')
    word_vectors.load_memory()

    def test_retrain_locked(self):
        train_save_path = 'data/test_data/test_load'

        with self.assertRaises(model_errors.ModelAlreadyTrainedError):
            params = ml.HyperParams(heads=5, learning_rate=1, dropout=0.3, optimizer=optim.Adadelta)

            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params, classifier_model=EntailmentTransformer)
            mike_net.train(epochs=100, print_every=10)

    def test_retrain_unlocked(self):
        train_save_path = 'data/test_data/test_retrain'

        with TestTeardown(train_save_path):
            params = ml.HyperParams(heads=5, learning_rate=1, dropout=0.3, optimizer=optim.Adadelta)

            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params, classifier_model=EntailmentTransformer)
            mike_net.train(epochs=5, print_every=10)

            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params, classifier_model=EntailmentTransformer)
            mike_net.unlock()
            mike_net.train(epochs=100, print_every=10)

            self.assertGreater(mike_net.history.accuracy[-1], 0.3)
            self.assertLess(mike_net.history.loss[-1], 3)
            self.assertLess(mike_net.info.runtime, 20)  # Red flag if it takes longer than 20 seconds.

    def test_patience_not_reset_when_loaded(self):
        train_save_path = 'data/test_data/test_load'

        params = HyperParams(heads=5, learning_rate=0.5, dropout=0.3, optimizer=optim.Adadelta,
                             patience=6, early_stopping_mode="minimum")
        mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                       hyper_parameters=params, classifier_model=NeuralNetwork,
                                       validation_data_loader=self.validation_loader)
        self.assertGreater(mike_net.early_stopping.trigger_times, 0)
        mike_net.unlock()
        self.assertGreater(mike_net.early_stopping.trigger_times, 0)

    def test_negative_patience_model(self):
        train_save_path = 'data/test_data/test_model'

        params = HyperParams(heads=5, learning_rate=0.5, dropout=0.3, optimizer=optim.Adadelta,
                             patience=-1, early_stopping_mode="minimum")

        with TestTeardown(train_save_path):
            with self.assertRaises(ValueError):
                StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                    hyper_parameters=params, classifier_model=NeuralNetwork,
                                    validation_data_loader=self.validation_loader)

    def test_loss_comparison_restored_STRICT(self):
        train_save_path = 'data/test_data/test_model'
        # Low patience and with strict early stopping, Likely to early stop.
        params = ml.HyperParams(learning_rate=0.5, patience=6, optimizer=optim.Adadelta, early_stopping_mode="strict")

        with TestTeardown(train_save_path):
            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params, classifier_model=EntailmentTransformer,
                                           validation_data_loader=self.train_loader)

            self.assertEqual(mike_net.early_stopping.loss_comparison, np.inf)
            mike_net.train(10)
            final_loss = mike_net.validation_history.loss[-1]
            self.assertAlmostEqual(mike_net.early_stopping.loss_comparison, final_loss)

            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params, classifier_model=EntailmentTransformer,
                                           validation_data_loader=self.train_loader)
            self.assertAlmostEqual(mike_net.early_stopping.loss_comparison, final_loss)

    def test_loss_comparison_restored_MIN(self):
        train_save_path = 'data/test_data/test_model'
        # Low patience and with strict early stopping, Likely to early stop.
        params = ml.HyperParams(learning_rate=0.5, patience=6, optimizer=optim.Adadelta, early_stopping_mode="minimum")

        with TestTeardown(train_save_path):
            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params, classifier_model=EntailmentTransformer,
                                           validation_data_loader=self.train_loader)

            self.assertEqual(mike_net.early_stopping.loss_comparison, np.inf)
            mike_net.train(10)
            final_loss = min(mike_net.validation_history.loss)
            self.assertAlmostEqual(mike_net.early_stopping.loss_comparison, final_loss)

            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params, classifier_model=EntailmentTransformer,
                                           validation_data_loader=self.train_loader)
            self.assertAlmostEqual(mike_net.early_stopping.loss_comparison, final_loss)

    def test_additional_info_init_exists(self):
        train_save_path = 'data/test_data/test_load'
        params = ml.HyperParams(heads=5, learning_rate=1, dropout=0.3, optimizer=optim.Adadelta)
        mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                       hyper_parameters=params, classifier_model=EntailmentTransformer)
        self.assertTrue(file_op.is_file(train_save_path + "/info.json"))
        self.assertGreater(mike_net.info.runtime, 0)
        self.assertEqual(32, mike_net.info["max_length"])

    def test_additional_info_init(self):
        train_save_path = 'data/test_data/test_model'
        # Low patience and with strict early stopping, Likely to early stop.
        params = ml.HyperParams(learning_rate=1, patience=1, optimizer=optim.Adadelta, early_stopping_mode="strict")

        with TestTeardown(train_save_path):
            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params, classifier_model=EntailmentTransformer,
                                           validation_data_loader=self.train_loader)
            self.assertTrue(file_op.is_file(train_save_path + "/info.json"))
            self.assertEqual(mike_net.info.runtime, 0)
            self.assertEqual(26, mike_net.info["max_length"])
            mike_net.train(200)

            self.assertGreater(mike_net.info.runtime, 0)
            self.assertEqual(26, mike_net.info["max_length"])
            self.assertGreater(mike_net.early_stopping.trigger_times, 0)

            with self.assertRaises(model_errors.ModelAlreadyTrainedError):
                mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                               hyper_parameters=params, classifier_model=EntailmentTransformer,
                                               validation_data_loader=self.train_loader)
                mike_net.unlock()
                mike_net.train(100)

    def test_runtime_saves_each_epoch(self):
        train_save_path = 'data/test_data/test_model'

        with TestTeardown(train_save_path):
            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           classifier_model=EntailmentTransformer,
                                           validation_data_loader=self.train_loader)
            self.assertEqual(mike_net.info.runtime, 0)
            mike_net.train(10)
            self.assertGreater(mike_net.info.runtime, 0)


if __name__ == '__main__':
    unittest.main()
