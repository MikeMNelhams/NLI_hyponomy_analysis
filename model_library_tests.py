import shutil
import unittest

import os.path
from dotenv import load_dotenv

from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import SNLI_DataLoader_Unclean
from NLI_hyponomy_analysis.data_pipeline import embeddings_library as embed
import model_library as ml
from models import StaticEntailmentNet, NeuralNetwork, EntailmentTransformer, HyperParams
import torch.optim as optim

import model_errors


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
    train_small_path = "data/snli_small/snli_small1_train.jsonl"
    train_loader = SNLI_DataLoader_Unclean(train_small_path)
    validation_loader = SNLI_DataLoader_Unclean(train_small_path)

    load_dotenv()

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
        params = HyperParams(heads=5, learning_rate=0.5, dropout=0.3, optimizer=optim.Adadelta,
                             patience=10, early_stopping_mode="minimum")

        mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path='data/test_data/test_load.pth',
                                       classifier_model=NeuralNetwork, hyper_parameters=params)

        loss, acc = mike_net.test(self.train_loader)
        print(f'LOSS: {loss}, ACC: {acc}')
        self.assertGreater(mike_net.history.accuracy[-1], 0.5)
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
            mike_net.train(epochs=100, print_every=10)

            mike_net = StaticEntailmentNet(self.word_vectors, self.train_loader, file_path=train_save_path + '.pth',
                                           hyper_parameters=params, classifier_model=EntailmentTransformer)
            mike_net.unlock()
            mike_net.train(epochs=100, print_every=10)

            self.assertGreater(mike_net.history.accuracy[-1], 0.3)
            self.assertLess(mike_net.history.loss[-1], 3)
            self.assertLess(mike_net.info.runtime, 20)  # Red flag if it takes longer than 20 seconds.


class ModelPropertyTesting(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
