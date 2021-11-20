import os.path
from abc import ABC, abstractmethod
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from prettytable import PrettyTable

from NLI_hyponomy_analysis.data_pipeline.file_operations import file_path_is_of_extension, JSON_writer
from NLI_hyponomy_analysis.data_pipeline.file_operations import is_file, file_path_without_extension

# CODE FROM: https://www.youtube.com/watch?v=U0s0f995w14&t=2494s
#   Paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf


def method_print_decorator(func: callable, symbol='-', number_of_symbol_per_line: int=40) -> callable:
    def wrapper(*args, **kwargs):
        print(symbol*number_of_symbol_per_line)
        func(*args, **kwargs)
        print(symbol*number_of_symbol_per_line)
    return wrapper


class EarlyStoppingTraining:
    """ https://clay-atlas.com/us/blog/2021/08/25/pytorch-en-early-stopping/ """
    modes = ("strict", "moving_average")

    def __init__(self, save_checkpoint: Callable, patience: int = 5, mode: str ="strict"):
        self.step = self.__select_measure(mode)

        self.patience = patience
        self.loss_comparison = 0
        self.trigger_times = 0

        # For saving checkpoints during each __call__
        self.save_checkpoint = save_checkpoint

    def __call__(self, loss: float) -> bool:
        return self.step(loss)

    def reset_validation_trigger(self) -> None:
        self.trigger_times = 0
        self.save_checkpoint()
        return None

    def __select_measure(self, mode) -> Callable[[float], bool]:
        self.assert_valid_mode(mode)

        if mode == "moving_average":
            return self.__moving_average

        return self.__strict

    def __strict(self, loss) -> bool:
        """ loss comparison = previous loss"""
        if loss > self.loss_comparison:
            self.trigger_times += 1
            print('Trigger times:', self.trigger_times)

            if self.trigger_times >= self.patience:
                print('Training Stopped Early!')
                return True
        else:
            self.reset_validation_trigger()

        self.loss_comparison = loss

        return False

    def __moving_average(self, loss) -> bool:
        """ loss comparison = (loss comparison + loss) / 2"""
        if loss > self.loss_comparison:
            self.trigger_times += 1
            print('Trigger times:', self.trigger_times)

            if self.trigger_times >= self.patience:
                print('Training Stopped Early!')
                return True
        else:
            self.reset_validation_trigger()

        self.loss_comparison = (self.loss_comparison + loss) / 2

        return False

    @staticmethod
    def assert_valid_mode(mode) -> None:
        if mode not in EarlyStoppingTraining.modes:
            raise TypeError
        return None


class HyperParams:
    def __init__(self, num_layers: int = 6, forward_expansion: int = 4, heads: int = 8, dropout: float = 0,
                 device="cuda", learning_rate: float = 0.1, optimizer=optim.Adadelta,
                 patience=5, early_stopping_mode="moving_average"):
        # General parameters
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Transformer parameters
        self.forward_expansion = forward_expansion
        self.heads = heads

        # Validation parameters
        self.patience = patience
        self.early_stopping_mode = early_stopping_mode

        # Read Only Fields
        self.__optimizer = optimizer
        self.__num_layers = num_layers

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def num_layers(self):
        return self.__num_layers


class History:
    """ Uses a csv file to save its loss and accuracy"""
    def __init__(self, file_path: str, precision: int = 4):
        self.__file_path = file_path
        self.__precision = precision

        self.__loss = []
        self.__accuracy = []

        if self.is_file():
            self.load()

    def __len__(self):
        return len(self.loss)

    @property
    def file_path(self):
        return self.__file_path

    @property
    def precision(self):
        return self.__precision

    @property
    def loss(self):
        return self.__loss

    @property
    def accuracy(self):
        return self.__accuracy

    def is_file(self) -> bool:
        return is_file(self.file_path, '.csv')

    def assert_not_empty(self) -> None:
        if len(self) == 0:
            print('No history has been provided! Loss/Accuracy empty!')
            raise ValueError
        return None

    def step(self, loss, accuracy) -> None:
        self.loss.append(round(loss, self.precision))
        self.accuracy.append(round(accuracy, self.precision))
        return None

    def save(self) -> None:
        self.assert_not_empty()
        print('\033[94mSaving Model History...\033[0m')  # BLUE TEXT
        data_to_write = pd.DataFrame({'loss': self.loss, 'accuracy': self.accuracy})
        # KNOWN INSPECTION BUG FOR TO_CSV()
        # https://stackoverflow.com/questions/68787744/
        #   pycharm-type-checker-expected-type-none-got-str-instead-when-using-pandas-d
        # noinspection PyTypeChecker
        data_to_write.to_csv(path_or_buf=self.file_path)
        return None

    def load(self) -> None:
        assert is_file(self.file_path, '.csv'), FileNotFoundError
        print('Loading Model History...')
        data_from_file = pd.read_csv(self.file_path)
        self.__loss = data_from_file['loss'].tolist()
        self.__accuracy = data_from_file['accuracy'].tolist()
        return None

    def plot_loss(self) -> None:
        assert self.is_file(), FileNotFoundError
        epoch_steps = range(len(self))
        plt.plot(epoch_steps, self.loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        return None

    def plot_accuracy(self) -> None:
        assert self.is_file(), FileNotFoundError
        epoch_steps = range(len(self))
        plt.plot(epoch_steps, self.accuracy)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()
        return None


class AdditionalInformation(JSON_writer):
    def __init__(self, file_path: str):
        super(AdditionalInformation, self).__init__(file_path)
        self.file_path = file_path

        self.data = {'total_runtime': 0}

    def add_runtime(self, runtime: float):
        self.data['total_runtime'] += runtime

    def reset_runtime(self):
        self.data['total_runtime'] = 0

    @property
    def runtime(self):
        return self.data['total_runtime']


class EntailmentSelfAttention(nn.Module):
    """ Implementation Code MODIFIED from: https://www.youtube.com/watch?v=U0s0f995w14&t=2494s,
            From paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"""

    def __init__(self, embed_size: int, heads: int):
        super(EntailmentSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dimension = embed_size // heads

        assert (self.head_dimension * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fully_connected_out = nn.Linear(embed_size, embed_size)

    def forward(self, values: np.array, keys: np.array, queries: np.array, mask):
        """ Multi-head concat Attention."""

        n = queries.shape[0]  # Batch Size
        value_len, key_len, query_len = values.shape[2], keys.shape[2], queries.shape[2]  # Max sequence len
        num_sentences = values.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Split pieces into self.heads pieces
        values = values.reshape(n, num_sentences, value_len, self.heads, self.head_dimension)
        keys = keys.reshape(n, num_sentences, key_len, self.heads, self.head_dimension)
        queries = queries.reshape(n, num_sentences, query_len, self.heads, self.head_dimension)

        # Multiply the queries with the keys. Q K^T = energy
        #   queries shape: (n, num_sent, query_len, heads, heads_dim)
        #   keys shape: (n, num_sent, keys_len, heads, heads_dim)
        # energy shape: (n, num_sent, heads, query_len, key_len)

        energy = torch.einsum("nsqhd,nskhd->nshqk", [queries, keys])

        if mask is not None:
            # Mask shape is         (n, seq_len, e, num_sent)
            # Energy shape is       (n, h, num_sent, seq_len, seq_len)
            # Desired mask shape is (n, h, num_sent, seq_len, seq_len)

            # Drop the Last index dimension, since it IS repeated along this dim. (embed size)
            mask_reshaped = mask[:, :, :, 0]

            # Repeat along final dim self.heads times
            # Permute the columns around.
            # Repeat along final dim value_len times.
            # TODO Rigorously PROVE this is correct
            mask_reshaped = mask_reshaped.unsqueeze(-1).expand(-1, -1, -1, self.heads)
            mask_reshaped = mask_reshaped.permute(0, 1, 3, 2)
            mask_reshaped = mask_reshaped.unsqueeze(-1).expand(-1, -1, -1, -1, value_len)

            energy = energy.masked_fill(mask_reshaped == 0, float("-1e20"))

        attention_softmax = torch.softmax(energy / (key_len ** (1 / 2)), dim=3)

        attention = torch.einsum("nshql,nslhd->nsqhd", [attention_softmax, values])
        # Back to the original shape
        attention = attention.reshape(n, num_sentences, value_len, self.heads * self.head_dimension)
        # Full equation: attention(Q, K, V) = softmax((Q K^T)/sqrt(d_k)) V
        #   attention_softmax shape: (n, heads, query_len, key_len)
        #   values shape: (n, value_len, heads, heads_dim)
        # attention shape: (n, query_len, heads, heads_dim) then flatten last two dimensions

        attention_out = self.fully_connected_out(attention)
        return attention_out


class EntailmentTransformerBlock(nn.Module):
    def __init__(self, embed_size: int, hyper_params: HyperParams=HyperParams()):
        super(EntailmentTransformerBlock, self).__init__()

        self.embed_size = embed_size
        self.hyper_params = hyper_params

        # Hyperparameter unpacking
        self.heads = self.hyper_params.heads
        self.forward_expansion = self.hyper_params.forward_expansion
        self.dropout = self.hyper_params.dropout

        # Model Architecture
        self.attention = EntailmentSelfAttention(embed_size=self.embed_size, heads=self.heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, self.forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(self.forward_expansion * embed_size, embed_size)
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(query + self.dropout(attention))  # Skip connection
        forward = self.feed_forward(x)
        output_layer = self.norm2(x + self.dropout(forward))  # Skip connection
        return output_layer


class EntailmentEncoder(nn.Module):
    def __init__(self, num_sentences: int, max_seq_len: int, embed_size: int=300,
                 hyper_parameters: HyperParams = HyperParams()):
        super(EntailmentEncoder, self).__init__()

        # Input shape: (batch_size, max_length, embed_size, num_sentences)
        self.num_sentences = num_sentences
        self.max_length = max_seq_len
        self.embed_size = embed_size
        self.hyper_params = hyper_parameters

        # Model structure
        self.position_embedding = nn.Embedding(self.max_length, self.embed_size)
        self.layers = nn.ModuleList(
            [
                EntailmentTransformerBlock(self.embed_size, self.hyper_params)
                for _ in range(self.hyper_params.num_layers)
            ]
        )
        self.dropout = nn.Dropout(self.hyper_params.dropout)

    def forward(self, x, mask):
        batch_size, num_sentences, sequence_length, _ = x.shape
        assert num_sentences == self.num_sentences

        # Positional encoding
        positions = torch.arange(0, sequence_length).expand(batch_size, sequence_length).to(self.hyper_params.device)
        positions_out = self.position_embedding(positions)
        positions_out = positions_out.unsqueeze(-1).expand(-1, -1, -1, num_sentences)  # Duplicate across num sentences
        positions_out = positions_out.permute(0, 3, 1, 2)
        out = self.dropout(x + positions_out)

        # Shape (b, max_len, e, num_sentences)
        for layer in self.layers:
            # Value, Key, Query, mask
            out = layer(out, out, out, mask)
        return out


class AbstractClassifierModel(ABC):
    """ Handles Model construction, loading, saving internally. You define train, test, validation, predict."""
    def __init__(self, train_data_loader, file_path: str, classifier_model, embed_size: int, input_shape,
                 num_classes: int, hyper_parameters: HyperParams = HyperParams()):
        # Essential objects
        self.data_loader = train_data_loader
        self.hyper_parameters = hyper_parameters

        # File I/O
        assert file_path_is_of_extension(file_path, '.pth'), FileNotFoundError
        self.file_path = file_path
        self.history_file_path = self._default_file_path_name + '_history.csv'
        self.history = History(self.history_file_path)  # For recording information

        self.info_file_path = self._default_file_path_name + '_info.json'
        self.info = AdditionalInformation(self.info_file_path)

        # Model shape // Input shape
        self.embed_size = embed_size
        self.input_shape = input_shape

        # Model structure
        self.num_classes = num_classes
        self.optimizer = hyper_parameters.optimizer
        self.max_length = input_shape[1]

        if self.is_file:
            self.load()
        else:
            self.model = classifier_model(self.input_shape,
                                          max_seq_len=self.max_length,
                                          hyper_parameters=hyper_parameters,
                                          number_of_output_classes=self.num_classes).to(hyper_parameters.device)
            self.optimizer = self.optimizer(self.model.parameters(), lr=self.hyper_parameters.learning_rate)

    @abstractmethod
    def train(self, epochs: int, criterion=nn.CrossEntropyLoss(), print_every: int = 1):
        raise NotImplementedError

    @abstractmethod
    def predict(self, batch: torch.Tensor, batch_mask: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def test(self, test_data_loader,
             test_batch_size: int=256, criterion=nn.CrossEntropyLoss()):
        raise NotImplementedError

    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        print("+---------------+------------+")
        return total_params

    @method_print_decorator
    def load(self) -> None:
        print('Loading model...')
        if not self.is_file:
            raise FileNotFoundError
        self.model = torch.load(self.file_path)
        print('Model loaded!')
        return None

    def save(self) -> None:
        self.save_checkpoint()
        self.save_model_training()
        return None

    def save_checkpoint(self) -> None:
        print('\033[96mSaving model checkpoint...\033[0m')  # CYAN TEXT
        torch.save(self.model, self.file_path)
        return None

    def save_model_training(self) -> None:
        print('\033[94mSaving model training...\033[0m')  # BLUE TEXT
        self.history.save()
        self.info.save()
        return None

    def _number_of_iterations_per_epoch(self, batch_size) -> int:
        num_iters = len(self.data_loader) // batch_size
        if batch_size > len(self.data_loader):
            num_iters = 1
        return num_iters

    @staticmethod
    def accuracy(x: torch.Tensor, y: torch.Tensor):
        correct = 0
        for x_row, y_row in zip(x, y):
            correct += int(x_row == y_row)
        accuracy = correct / int(x.shape[0])
        return accuracy

    @property
    def is_file(self) -> bool:
        return os.path.isfile(self.file_path)

    @staticmethod
    @method_print_decorator
    def print_available_devices() -> None:
        print(f'{torch.cuda.device_count()} devices available')
        device_indices = list(range(torch.cuda.device_count()))
        for device_idx in device_indices:
            print('Device:', device_idx)
            print('Device Name:', torch.cuda.get_device_name(device_idx))
        return None

    @property
    def _default_file_path_name(self):
        return file_path_without_extension(self.file_path)

    @staticmethod
    def _minibatch_predictions(x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(x, dim=1)


def main():
    pass


if __name__ == "__main__":
    main()
