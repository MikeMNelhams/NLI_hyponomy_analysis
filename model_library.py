from __future__ import annotations
import os.path
from abc import ABC, abstractmethod
from typing import Callable, Dict, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from prettytable import PrettyTable

import NLI_hyponomy_analysis.data_pipeline.file_operations as file_op

from NLI_hyponomy_analysis.data_pipeline.NLI_data_handling import NLI_DataLoader_abc

from sklearn.metrics import confusion_matrix, accuracy_score

# CODE FROM: https://www.youtube.com/watch?v=U0s0f995w14&t=2494s
#   Paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf


def method_print_decorator(func: callable, symbol='-', number_of_symbol_per_line: int = 40) -> callable:
    def wrapper(*args, **kwargs):
        print(symbol * number_of_symbol_per_line)
        func(*args, **kwargs)
        print(symbol * number_of_symbol_per_line)

    return wrapper


class Checkpoint:
    def __init__(self, model, optimizer: optim.Optimizer):
        self.state_dict = model.state_dict()
        self.optimizer = optimizer.state_dict()

    def __getstate__(self):
        return {"state_dict": self.state_dict, "optimizer": self.optimizer}

    def __setstate__(self, state):
        self.optimizer = state["optimizer"]
        self.state_dict = state["state_dict"]

    def __repr__(self):
        return str(self.__getstate__())

    def save(self, file_path: str) -> None:
        state = self.__getstate__()
        torch.save(state, file_path)
        return None

    def safe_save(self, file_path: str) -> None:
        if file_op.is_file(file_path):
            raise FileExistsError
        self.save(file_path)
        return None

    def apply(self, model, optimizer: optim.Optimizer) -> (Any, optim.Optimizer):
        """ Apply the checkpoint saved values to instances of models or optimizers """
        model.load_state_dict(self.state_dict)

        if self.is_legacy_checkpoint:
            self.optimizer = optimizer.state_dict()
        else:
            optimizer.load_state_dict(self.optimizer)
        return model, optimizer

    @classmethod
    def load_checkpoint(cls, file_path: str, device: torch.device=torch.device("cpu")) -> Checkpoint:
        if not file_op.is_file(file_path):
            raise FileNotFoundError
        state = torch.load(file_path, map_location=device)
        checkpoint = cls.__new__(cls)
        if not isinstance(state, dict) or "state_dict" not in state or "optimizer" not in state:
            checkpoint.state_dict = state
            checkpoint.optimizer = None
            return checkpoint

        checkpoint.state_dict = state["state_dict"]
        checkpoint.optimizer = state["optimizer"]
        return checkpoint

    @property
    def is_legacy_checkpoint(self):
        """Old checkpoints were stored purely as state dicts, not {state_dict, optimizer}"""
        return self.optimizer is None


class EarlyStoppingTraining:
    """ https://clay-atlas.com/us/blog/2021/08/25/pytorch-en-early-stopping/ """
    modes = ("strict", "moving_average", "minimum", "none")

    def __init__(self, save_checkpoint: Callable, file_path: str,
                 patience: int = 5, validation_history: Union[None, History] = None, mode: str = "minimum"):
        self.step = self.__select_measure(mode)

        self.__assert_valid_patience(patience)

        self.patience = patience
        self.triggers_file_writer = file_op.TextWriterSingleLine(file_path)

        val_history = validation_history if len(validation_history) != 0 else None
        self.loss_comparison = self.__calculate_loss_comparison(mode, val_history)

        self.trigger_times = 0
        if self.triggers_file_writer.file_exists:
            trigger_times = self.triggers_file_writer.load_safe()
            if trigger_times != '':
                self.trigger_times = int()

        # For saving checkpoints during each __call__
        self.save_checkpoint = save_checkpoint

    def __call__(self, loss: float) -> bool:
        out = self.step(loss)
        self.save_trigger_times()
        return out

    def reset_validation_trigger(self) -> None:
        self.trigger_times = 0
        self.save_checkpoint()
        return None

    def save_trigger_times(self) -> None:
        self.triggers_file_writer.save(self.trigger_times)
        return None

    def __select_measure(self, mode) -> Callable[[float], bool]:
        self.assert_valid_mode(mode)

        if mode == "moving_average":
            return self.__moving_average

        if mode == "minimum":
            return self.__minimum

        if mode == "none":
            return self.__none

        if mode == "strict":
            return self.__strict

        return self.__minimum

    def __calculate_loss_comparison(self, mode, validation_history: Union[None, History] = None):
        self.assert_valid_mode(mode)

        if validation_history is None:
            return np.inf

        if mode == "moving_average":
            average = 0
            for val_loss in validation_history.loss:
                average = (val_loss + average) / 2
            return average

        if mode == "minimum":
            return min(validation_history.loss)

        if mode == "strict":
            return validation_history.loss[-1]

        return np.inf

    def __strict(self, loss) -> bool:
        """ loss comparison = previous loss"""
        if loss >= self.loss_comparison:
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
        if loss >= self.loss_comparison:
            self.trigger_times += 1
            print('Trigger times:', self.trigger_times)

            if self.trigger_times >= self.patience:
                print('Training Stopped Early!')
                return True
        else:
            self.reset_validation_trigger()

        self.loss_comparison = (self.loss_comparison + loss) / 2

        return False

    def __minimum(self, loss) -> bool:
        """ loss comparison = global minimum loss"""
        print(f'LOSS: {loss}, LC: {self.loss_comparison}')
        if loss >= self.loss_comparison:
            self.trigger_times += 1
            print('Trigger times:', self.trigger_times)

            if self.trigger_times >= self.patience:
                print('Training Stopped Early!')
                return True
        else:
            self.reset_validation_trigger()
            self.loss_comparison = loss

        return False

    def __none(self, loss) -> bool:
        self.reset_validation_trigger()
        return False

    @staticmethod
    def assert_valid_mode(mode) -> None:
        if mode not in EarlyStoppingTraining.modes:
            raise ValueError
        return None

    @staticmethod
    def __assert_valid_patience(patience: int) -> None:
        if type(patience) != int:
            raise TypeError
        if patience < 0:
            raise ValueError
        return None


class Regularisation:
    def __init__(self, l1=0, l2=0):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, model):
        return self.loss(model)

    def loss(self, model) -> float:
        total_l_loss = 0
        if self.l1 > 0:
            total_l_loss += self.__l_norm_loss(1, self.l1, model)
        if self.l2 > 0:
            total_l_loss += self.__l_norm_loss(2, self.l2, model)

        return total_l_loss

    @staticmethod
    def __l_norm_loss(l_norm, l_coefficient, model) -> float:

        if l_norm == 1:
            norm = sum(p.abs().sum() for p in model.parameters())
        else:
            norm = sum(p.pow(float(l_norm)).sum() for p in model.parameters())

        l_loss = l_coefficient * norm

        return l_loss


class HyperParams:
    def __init__(self, num_layers: int = 6, forward_expansion: int = 4, heads: int = 5, dropout: float = 0,
                 device="cuda", learning_rate: float = 0.1, optimizer=optim.Adadelta, regularisation=Regularisation(),
                 patience=5, early_stopping_mode="moving_average"):
        # General parameters
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.regularisation = regularisation
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Transformer parameters
        self.forward_expansion = forward_expansion
        self.heads = heads

        # Validation parameters
        self.patience = patience
        self.early_stopping_mode = early_stopping_mode

        # Read Only Fields
        # self.__optimizer = optimizer
        self.__optimizer = optimizer
        self.__num_layers = num_layers

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def num_layers(self):
        return self.__num_layers


class Metrics:
    def __init__(self, predictions: torch.Tensor, labels: torch.Tensor):
        self.confusion_mtrx = confusion_matrix(predictions, labels)
        self.__accuracy = accuracy_score(predictions, labels)

    def recall(self) -> float:
        recall = np.diag(self.confusion_mtrx) / np.sum(self.confusion_mtrx, axis=1)
        recall_mean = float(np.mean(recall))
        return recall_mean

    def precision(self) -> float:
        precision = np.diag(self.confusion_mtrx) / np.sum(self.confusion_mtrx, axis=0)
        precision_mean = float(np.mean(precision))
        return precision_mean

    def f1_score(self) -> float:
        precision = self.precision()
        recall = self.recall()

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def accuracy(self) -> float:
        return self.__accuracy


class MetricEvaluator:
    metric_scores = ("recall", "precision", "f1_score", "accuracy")

    class InvalidMetricError(Exception):
        def __init__(self, metric_name: str):
            message = f"The given metric: {metric_name} is invalid. Try one of {MetricEvaluator.metric_scores}."
            super().__init__(message)

    def __init__(self, *metric_names):
        for metric in metric_names:
            self.__assert_valid_metric(metric)

        self.metrics_to_track = metric_names

    def __len__(self):
        return len(self.metrics_to_track)

    def __str__(self):
        return str(self.metrics_to_track)

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        return self.evaluate(predictions, labels)

    def evaluate(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        metrics = Metrics(predictions, labels)

        metric_functions = [self.__get_metric(metric, metrics) for metric in self.metrics_to_track]
        metrics_evaluated = {metric: metric_function()
                             for metric_function, metric in zip(metric_functions, self.metrics_to_track)}
        return metrics_evaluated

    @staticmethod
    def __get_metric(metric_func_name: str, metrics_instance: Metrics) -> Callable:
        metric_function = getattr(metrics_instance, metric_func_name)
        return metric_function

    @staticmethod
    def __assert_valid_metric(metric_name: str) -> None:
        assert metric_name in MetricEvaluator.metric_scores, MetricEvaluator.InvalidMetricError
        return None

    @staticmethod
    def print_all_metric_types() -> None:
        print(MetricEvaluator.metric_scores)
        return None


class History:
    """ Uses a csv file to write its loss and accuracy"""

    def __init__(self, file_path: str, decimal_places: int = -1, label="training"):
        self.__file_path = file_path
        self.__decimal_places = decimal_places

        self.__loss = []
        self.__accuracy = []

        self.__additional_metrics = None

        if self.is_file():
            self.load()

        self.label = label

    def __len__(self):
        return len(self.loss)

    @property
    def file_path(self):
        return self.__file_path

    @property
    def decimal_places(self):
        return self.__decimal_places

    @property
    def loss(self):
        return self.__loss

    @property
    def tracking_additional_metrics(self):
        return self.__additional_metrics is not None

    @property
    def accuracy(self):
        return self.__accuracy

    def is_file(self) -> bool:
        return file_op.is_file(self.file_path)

    def assert_not_empty(self) -> None:
        if len(self) == 0:
            print('No history has been provided! Loss/Accuracy empty!')
            raise ValueError
        return None

    def step(self, loss, accuracy, additional_metrics: Dict[str, float] = None) -> None:
        rounded_loss = round(loss, self.decimal_places) if self.decimal_places != -1 else loss
        rounded_accuracy = round(accuracy, self.decimal_places) if self.decimal_places != -1 else accuracy
        self.loss.append(rounded_loss)
        self.accuracy.append(rounded_accuracy)

        if additional_metrics is not None:
            for key, value in additional_metrics.items():
                self.__additional_metrics[key].append(value)
        return None

    def track_metrics(self, metric_evaluator: MetricEvaluator) -> None:
        self.__additional_metrics = {metric: [np.nan for _ in range(len(self.loss))]
                                     for metric in metric_evaluator.metrics_to_track}
        return None

    def save(self) -> None:
        self.assert_not_empty()
        print('\033[94mSaving Model History...\033[0m')  # BLUE TEXT
        data_to_write = pd.DataFrame({'loss': self.loss, 'accuracy': self.accuracy})

        if self.tracking_additional_metrics:
            for key, value in self.__additional_metrics.items():
                data_to_write[key] = value

        # KNOWN INSPECTION BUG FOR TO_CSV()
        # https://stackoverflow.com/questions/68787744/
        #   pycharm-type-checker-expected-type-none-got-str-instead-when-using-pandas-d
        # noinspection PyTypeChecker
        data_to_write.to_csv(path_or_buf=self.file_path)
        return None

    def load(self) -> None:
        assert file_op.is_file(self.file_path), FileNotFoundError
        print('Loading Model History...')
        data_from_file = pd.read_csv(self.file_path)
        self.__loss = data_from_file['loss'].tolist()
        self.__accuracy = data_from_file['accuracy'].tolist()

        if self.tracking_additional_metrics:
            data_from_file.pop('loss')
            data_from_file.pop('accuracy')

            for key, value in data_from_file.items():
                self.__additional_metrics[key] = data_from_file[key].tolist()

        return None

    def plot_loss(self, axes=None, title='') -> plt.axes:
        assert self.is_file(), FileNotFoundError
        epoch_steps = range(len(self))

        figure_loss = self.loss.copy()

        for i, value in enumerate(self.loss):
            figure_loss[i] = round(value, 3)

        ax = axes
        if axes is None:
            fig, ax = plt.subplots()

        ax.plot(epoch_steps, figure_loss, label=self.label)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

        ax.legend()

        return ax

    def plot_accuracy(self, axes=None, title='', fontsize: int=20) -> plt.axes:
        assert self.is_file(), FileNotFoundError
        epoch_steps = range(len(self))

        figure_acc = self.accuracy.copy()

        for i, value in enumerate(self.accuracy):
            figure_acc[i] = round(value, 3)

        ax = axes
        if axes is None:
            fig, ax = plt.subplots()

        ax.plot(epoch_steps, figure_acc, label=self.label)
        ax.set_title(title)
        ax.titlesize = fontsize
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')

        ax.legend()

        return ax


class AdditionalInformation:
    def __init__(self, file_path: str):
        self.file_path = file_path

        self.__dict_writer = file_op.DictWriter(self.file_path)

        self.data = {'total_runtime': 0, "max_length": 0}

        if self.__dict_writer.file_exists:
            self.data = self.__dict_writer.load()

    def __setitem__(self, key: str, value):
        self.data[key] = value
        return None

    def __getitem__(self, key):
        return self.data[key]

    def __delitem__(self, key):
        if key in self.protected_keys:
            raise KeyError
        del self.data[key]
        return None

    @property
    def protected_keys(self):
        """These cannot be deleted"""
        return ["total_runtime", "max_length"]

    @property
    def is_file(self):
        return self.__dict_writer.file_exists

    def add_runtime(self, runtime: float):
        self.data['total_runtime'] += runtime

    def reset_runtime(self):
        self.data['total_runtime'] = 0

    @property
    def runtime(self):
        return self.data['total_runtime']

    def save(self) -> None:
        self.__dict_writer.save(self.data)
        return None

    def load(self) -> dict:
        return self.__dict_writer.load()


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
        del mask_reshaped

        divisor = key_len ** (1 / 2)
        attention_softmax = torch.softmax(energy / divisor, dim=3)

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
    def __init__(self, embed_size: int, hyper_params: HyperParams = HyperParams()):
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
    def __init__(self, num_sentences: int, max_seq_len: int, embed_size: int = 300,
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


class AbstractClassifierModel(ABC):
    """ Handles Model construction, loading, saving internally. You define train, test, validation, predict."""

    def __init__(self, train_data_loader: NLI_DataLoader_abc, file_path: str, classifier_model, embed_size: int,
                 input_shape, num_classes: int,
                 hyper_parameters: HyperParams = HyperParams()):
        # Essential objects
        self.data_loader = train_data_loader
        self.hyper_parameters = hyper_parameters

        self.metric_evaluator = None

        # File I/O
        assert file_op.file_path_is_of_extension(file_path, '.pth'), FileNotFoundError

        self._file_dir_path = file_op.file_path_without_extension(file_path) + '/'
        self.model_save_path = self._file_dir_path + 'model.pth'

        self.__make_dir()

        self.history_file_path = self._file_dir_path + 'history.csv'
        self.history = History(self.history_file_path, label="Training")  # For recording information

        self.info_file_path = self._file_dir_path + 'info.json'
        self.info = AdditionalInformation(self.info_file_path)

        # Model shape // Input shape
        self.embed_size = embed_size
        self.input_shape = input_shape

        # Model structure
        self.num_classes = num_classes
        self.max_length = input_shape[1]

        # Construct the model

        if not self.info.is_file:
            self.info["max_length"] = self.max_length
            self.info.save()  # This is important for init the model.

        self.regularisation = hyper_parameters.regularisation
        self.model = None
        self.optimizer = hyper_parameters.optimizer
        self._construct_model(classifier_model=classifier_model)

    def __make_dir(self) -> None:
        if file_op.is_dir(self._file_dir_path):
            return None

        file_op.make_dir(self._file_dir_path)
        return None

    def _construct_model(self, classifier_model: Callable) -> None:
        self.model = classifier_model(self.input_shape,
                                      max_seq_len=self.max_length,
                                      hyper_parameters=self.hyper_parameters,
                                      number_of_output_classes=self.num_classes).to(self.hyper_parameters.device)
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.hyper_parameters.learning_rate)
        if self.is_file:
            self.load()

        return None

    @abstractmethod
    def train(self, epochs: int, criterion=nn.CrossEntropyLoss(), print_every: int = 1):
        raise NotImplementedError

    @abstractmethod
    def predict(self, batch: torch.Tensor, batch_mask: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def test(self, test_data_loader: NLI_DataLoader_abc,
             test_batch_size: int = 256, criterion=nn.CrossEntropyLoss()):
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

        final_checkpoint = Checkpoint.load_checkpoint(self.model_save_path, device=self.hyper_parameters.device)
        final_checkpoint.apply(self.model, self.optimizer)

        self.model.eval()
        self.model = self.model.to(self.hyper_parameters.device)
        print('Model loaded!')
        return None

    def save(self) -> None:
        self.save_checkpoint()
        self.save_model_history()
        return None

    def save_checkpoint(self) -> None:
        print('\033[96mSaving model checkpoint...\033[0m')  # CYAN TEXT
        checkpoint = Checkpoint(self.model, self.optimizer)
        checkpoint.save(self.model_save_path)
        return None

    def save_model_history(self) -> None:
        print('\033[94mSaving model history...\033[0m')  # BLUE TEXT
        self.history.save()
        self.info.save()
        return None

    def plot_accuracy(self, title='') -> plt.axes:
        ax = self.history.plot_accuracy(title=title)
        return ax

    def plot_loss(self, title='') -> plt.axes:
        ax = self.history.plot_loss(title=title)
        return ax

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
        return os.path.isfile(self.model_save_path)

    @staticmethod
    @method_print_decorator
    def print_available_devices() -> None:
        print(f'{torch.cuda.device_count()} devices available')
        device_indices = list(range(torch.cuda.device_count()))
        for device_idx in device_indices:
            print('Device:', device_idx)
            print('Device Name:', torch.cuda.get_device_name(device_idx))
        return None

    @staticmethod
    def _minibatch_predictions(x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(x, dim=1)


def main():
    pass


if __name__ == "__main__":
    main()
