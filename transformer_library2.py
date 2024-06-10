import warnings
from typing import Any

import os.path

from data_pipeline.file_operations import is_file, file_path_without_extension, file_path_is_of_extension
from data_pipeline import SNLI_data_handling

import pandas as pd

from prettytable import PrettyTable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


# CODE FROM: https://www.youtube.com/watch?v=U0s0f995w14&t=2494s
#   Paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
class ModelAlreadyTrainedError(Exception):
    def __init__(self, file_path: str):
        super().__init__(f'Model cannot train, since it already has been trained and saved to: {file_path}')


class HyperParams:
    def __init__(self, num_layers: int = 6, forward_expansion: int = 4, heads: int = 8, dropout: float = 0,
                 device="cuda", batch_size: int = 256, learning_rate: float = 0.1):
        self.num_layers = num_layers
        self.forward_expansion = forward_expansion
        self.heads = heads
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")


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
        print('Saving Model History...')
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

        # Hyper Parameter unpacking
        self.heads = self.hyper_params.heads
        self.forward_expansion = self.hyper_params.forward_expansion
        self.dropout = self.hyper_params.dropout

        # Model structure
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
        out = self.norm2(x + self.dropout(forward))  # Skip connection
        return out


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
        positions = torch.arange(0, sequence_length).expand(batch_size, sequence_length)
        positions_out = self.position_embedding(positions)
        positions_out = positions_out.unsqueeze(-1).expand(-1, -1, -1, num_sentences)  # Duplicate across num sentences
        positions_out = positions_out.permute(0, 3, 1, 2)
        out = self.dropout(x + positions_out)

        # Shape (b, max_len, e, num_sentences)
        for layer in self.layers:
            # Value, Key, Query, mask
            out = layer(out, out, out, mask)
        return out


class EntailmentTransformer(nn.Module):
    def __init__(self, data_shape, max_seq_len: int, number_of_output_classes=3,
                 hyper_parameters: HyperParams = HyperParams()):
        super(EntailmentTransformer, self).__init__()

        # Input shape: (batch_size, num_sentences, max_length, embed_size)
        _, self.num_sentences, self.max_length, self.embed_size = data_shape
        print('Batch Default Shape:', data_shape)
        self.hyper_params = hyper_parameters
        self.hyper_params.embed_size = self.embed_size

        self.encoder_flattened_size = self.num_sentences * max_seq_len * self.embed_size

        # Model structure
        self.encoder = EntailmentEncoder(self.num_sentences, max_seq_len,
                                         embed_size=self.embed_size, hyper_parameters=self.hyper_params)
        self.fc1 = nn.Linear(self.encoder_flattened_size, max_seq_len, bias=True)
        self.fc2 = nn.Linear(max_seq_len, 75, bias=True)
        self.fc_out = nn.Linear(75, number_of_output_classes, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        # Input shape: (batch_size, num_sentences, embed_size, max_length)
        batch_size = x.shape[0]
        x = self.encoder(x, mask)
        x = x.masked_fill(mask == 0, 1e-20)
        x = x.reshape(batch_size, self.encoder_flattened_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class NeuralNetwork(nn.Module):
    def __init__(self, data_shape, max_seq_len: int, number_of_output_classes=3,
                 hyper_parameters: HyperParams = HyperParams()):
        super(NeuralNetwork, self).__init__()
        self.hyper_parameters = hyper_parameters

        # Input shape: n, num_sentences, max_seq_len, embed_size
        _, self.num_sentences, self.max_length, self.embed_size = data_shape
        self.encoder_flattened_size = self.num_sentences * max_seq_len * self.embed_size

        self.fc1 = nn.Linear(self.encoder_flattened_size, 1000, bias=True)
        self.fc2 = nn.Linear(1000, 75, bias=True)
        self.fc_out = nn.Linear(75, number_of_output_classes, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        # Input shape: (batch_size, num_sentences, embed_size, max_length)
        batch_size = x.shape[0]
        x = x.masked_fill(mask == 0, 1e-20)
        x = x.reshape(batch_size, self.encoder_flattened_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class EntailmentNet:
    def __init__(self, word_vectors, data_loader: SNLI_data_handling.SNLI_DataLoader, path: str,
                 hyper_parameters: HyperParams = HyperParams(), classifier_model=EntailmentTransformer):
        self.data_loader = data_loader
        self.word_vectors = word_vectors
        self.hyper_parameters = hyper_parameters
        assert file_path_is_of_extension(path, '.pth'), FileNotFoundError
        self.file_path = path
        self.history_file_path = self.__history_save_path()

        self.history = History(self.history_file_path)

        self.device = hyper_parameters.device

        self.batch_size = hyper_parameters.batch_size
        self.embed_size = word_vectors.d_emb
        self.num_sentences = data_loader.num_sentences
        self.input_shape = (self.batch_size, self.num_sentences,
                            data_loader.max_words_in_sentence_length, self.embed_size)

        # Model structure
        self.num_classes = 3  # Definition of problem means this is always 3 (4 if you want a 'not sure')

        if self.is_file:
            self.load_model()
        else:
            self.model = classifier_model(self.input_shape,
                                                max_seq_len=data_loader.max_words_in_sentence_length,
                                                hyper_parameters=hyper_parameters,
                                                number_of_output_classes=self.num_classes).cuda()
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.hyper_parameters.learning_rate)

    def train(self, epochs: int, criterion=nn.CrossEntropyLoss(), print_every: int = 1):
        if self.is_file:
            raise ModelAlreadyTrainedError(self.file_path)

        number_of_iterations_per_epoch = len(self.data_loader) // self.batch_size
        if self.batch_size > len(self.data_loader):
            number_of_iterations_per_epoch = 1
            self.batch_size = len(self.data_loader)
            self.hyper_parameters.batch_size = self.batch_size
        total_steps = epochs * number_of_iterations_per_epoch
        for epoch in range(epochs):
            running_loss = 0.0
            running_accuracy = 0.0
            for i in range(number_of_iterations_per_epoch):
                percentage_complete = round((100 * (epoch * number_of_iterations_per_epoch + i))/total_steps, 2)
                print(f'Training batch {i} of {number_of_iterations_per_epoch}. {percentage_complete}% done')

                loss, accuracy = self.__train_batch(criterion)

                # print statistics
                running_loss += loss.item()
                running_accuracy += accuracy
                if i % print_every == print_every - 1:
                    print('[%d, %5d] loss: %.4f \t accuracy: %.2f' %
                          (epoch + 1, i + 1, float(loss), 100 * accuracy))
                print('-' * 20)
            running_accuracy = running_accuracy / number_of_iterations_per_epoch
            running_loss = running_loss / number_of_iterations_per_epoch
            self.history.step(float(running_loss), running_accuracy)

        print('Finished Training.')

        self.save_model()
        self.history.save()
        return None

    def __train_batch(self, criterion=nn.CrossEntropyLoss()) -> Any:
        batch = self.data_loader.load_clean_batch_random(self.batch_size)

        inputs, masks = batch.to_tensors(self.word_vectors, pad_value=-1e-20)

        labels = batch.labels_encoding
        del batch

        # Put all on GPU
        inputs = inputs.cuda()
        masks = masks.cuda()
        labels = labels.cuda()

        # Zero the parameter gradients.
        self.optimizer.zero_grad()

        # Forward -> backward -> optimizer
        outputs = self.model(inputs, masks)
        predictions = self.__minibatch_predictions(outputs)
        print('MODEL OUTPUT:\n' + '-'*20)
        print(predictions)
        print(labels)
        print('-' * 30)

        loss = criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy

    def predict(self, batch: torch.Tensor, batch_mask: torch.Tensor = None) -> torch.Tensor:
        # Switch to eval mode, then switch back at the end.
        self.model.eval()
        self.hyper_parameters.dropout = 0

        if batch_mask is None:
            prediction = self.model(batch)
        else:
            prediction = self.model(batch, batch_mask)

        prediction = torch.argmax(prediction, dim=1)
        self.model.train()
        return prediction

    def test(self, test_data_loader: SNLI_data_handling.SNLI_DataLoader, test_batch_size: int=256, criterion=nn.CrossEntropyLoss()):
        if not self.is_file:
            self.__warn_not_trained()

        self.model.eval()
        max_batch_size = min(len(test_data_loader), test_batch_size)

        number_of_test_iterations = len(test_data_loader) // max_batch_size

        number_guessed_correctly = 0
        loss = 0
        for i in range(number_of_test_iterations):
            test_data = test_data_loader.load_clean_batch_sequential(batch_size=max_batch_size)
            lines, masks = test_data.to_tensors(self.word_vectors, pad_value=1e-20)
            labels = test_data.labels_encoding
            outputs = self.model(lines, masks)

            for label, prediction in zip(labels, torch.argmax(outputs, dim=1)):
                number_guessed_correctly += int(label == prediction)
                loss += criterion(outputs, labels)

        accuracy = number_guessed_correctly / (number_of_test_iterations * max_batch_size)
        print(f'Total loss: {round(float(loss), 4)}. Total accuracy: {round(accuracy * 100, 2)}%')
        self.model.train()
        return loss, accuracy

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
        return total_params

    @property
    def is_file(self) -> bool:
        return os.path.isfile(self.file_path)

    def load_model(self) -> None:
        print('-'*20)
        print('Loading model...')
        if not self.is_file:
            raise FileNotFoundError
        self.model = torch.load(self.file_path)
        print('Model loaded!')
        print('-' * 20)
        return None

    def save_model(self) -> None:
        print('Saving model...')
        torch.save(self.model, self.file_path)
        return None

    def __history_save_path(self):
        return file_path_without_extension(self.file_path) + '_history.csv'

    @staticmethod
    def print_available_devices() -> None:
        print('-'*30)
        print(f'{torch.cuda.device_count()} devices available')
        device_indices = list(range(torch.cuda.device_count()))
        for device_idx in device_indices:
            print('Device:', device_idx)
            print('Device Name:', torch.cuda.get_device_name(device_idx))
        print('-' * 30)
        return None

    @staticmethod
    def __minibatch_predictions(x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(x, dim=1)

    @staticmethod
    def accuracy(x: torch.Tensor, y: torch.Tensor):
        correct = 0
        for x_row, y_row in zip(x, y):
            correct += int(x_row == y_row)
        accuracy = correct / int(x.shape[0])
        return accuracy

    @staticmethod
    def __warn_not_trained() -> None:
        warnings.warn('WARNING: The model is not trained yet!')
        return None


def main():
    pass

if __name__ == "__main__":
    main()
