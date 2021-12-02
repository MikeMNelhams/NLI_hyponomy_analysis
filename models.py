import time
import torch
from torch import nn

from model_errors import ModelAlreadyTrainedError, ModelIsNotValidatingError
from model_library import HyperParams, EntailmentEncoder, AbstractClassifierModel, History, EarlyStoppingTraining


class NeuralNetwork(nn.Module):
    def __init__(self, data_shape, max_seq_len: int, number_of_output_classes=3,
                 hyper_parameters: HyperParams = HyperParams()):
        super(NeuralNetwork, self).__init__()
        self.hyper_parameters = hyper_parameters

        # Input shape: batch_size, num_sentences, max_seq_len, embed_size
        # Data shape: num_sentences, max_seq_len, embed_size
        self.num_sentences, self.max_length, self.embed_size = data_shape
        self.encoder_flattened_size = self.num_sentences * max_seq_len * self.embed_size

        self.fc1 = nn.Linear(self.encoder_flattened_size, 200, bias=True)
        self.fc2 = nn.Linear(200, 50, bias=True)
        self.fc_out = nn.Linear(50, number_of_output_classes, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        x = x.masked_fill(mask == 0, 1e-20)

        # Input shape: (batch_size, num_sentences, embed_size, max_length)
        batch_size = x.shape[0]

        x = x.reshape(batch_size, self.encoder_flattened_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class EntailmentTransformer(nn.Module):
    def __init__(self, data_shape, max_seq_len: int, number_of_output_classes=3,
                 hyper_parameters: HyperParams = HyperParams()):
        super(EntailmentTransformer, self).__init__()

        # Input shape: (batch_size, num_sentences, max_length, embed_size)
        # Data shape: num_sentences, max_seq_len, embed_size
        self.num_sentences, self.max_length, self.embed_size = data_shape
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


class StaticEntailmentNet(AbstractClassifierModel):
    batch_load_configs = ("sequential", "random")

    def __init__(self, word_vectors, train_data_loader, file_path: str,
                 hyper_parameters: HyperParams = HyperParams(), classifier_model=EntailmentTransformer,
                 validation_data_loader=None):

        model_is_validating = validation_data_loader is not None
        max_length = train_data_loader.max_words_in_sentence_length

        if model_is_validating:
            max_length = max(max_length, validation_data_loader.max_words_in_sentence_length)

        embed_size = word_vectors.d_emb
        # TODO make this dynamic based on data loader.
        num_classes = 4  # 0, 1, 2, 3. Entailment, Neutral, Contradiction, -

        input_shape = (train_data_loader.num_sentences, max_length, embed_size)

        super(StaticEntailmentNet, self).__init__(train_data_loader=train_data_loader, file_path=file_path,
                                                  classifier_model=classifier_model, hyper_parameters=hyper_parameters,
                                                  embed_size=embed_size, input_shape=input_shape,
                                                  num_classes=num_classes)

        # Essential objects
        self.word_vectors = word_vectors

        # Model structure
        self.num_sentences = train_data_loader.num_sentences
        self.max_length = max_length

        # Validation
        self.model_is_validating = model_is_validating
        if self.model_is_validating:
            self.__validation_data_loader = validation_data_loader
            self.__validation_save_path = self._default_file_path_name + '_validation_history.csv'
            self.validation_history = History(self.__validation_save_path)
            self.early_stopping = EarlyStoppingTraining(save_checkpoint=self.save_checkpoint,
                                                        patience=self.hyper_parameters.patience,
                                                        mode=self.hyper_parameters.early_stopping_mode)

    def train(self, epochs: int, batch_size: int=256, criterion=nn.CrossEntropyLoss(), batch_loading_mode="sequential",
              print_every: int = 1):
        training_start_time = time.perf_counter()

        batch_loading_function = self.__load_clean_batch(batch_loading_mode)

        if self.is_file:
            raise ModelAlreadyTrainedError(self.file_path)

        number_of_iterations_per_epoch = self._number_of_iterations_per_epoch(batch_size=batch_size)

        total_steps = epochs * number_of_iterations_per_epoch
        for epoch in range(epochs):
            running_loss = 0.0
            running_accuracy = 0.0
            for i in range(number_of_iterations_per_epoch):
                percentage_complete = round((100 * (epoch * number_of_iterations_per_epoch + i)) / total_steps, 2)
                should_print = i % print_every == print_every - 1
                if should_print:
                    print(f'Training batch: {i} of {number_of_iterations_per_epoch}.\t {percentage_complete}% done')
                loss, accuracy = self.__train_batch(batch_loader=batch_loading_function,
                                                            batch_size=batch_size, criterion=criterion)

                # print statistics
                batch_loss = loss.item()
                running_loss += batch_loss
                running_accuracy += accuracy
                if should_print:
                    self.__print_step(epoch=epoch, batch_step=i, loss=batch_loss, accuracy=accuracy)
                    print('-' * 50)

            running_accuracy = running_accuracy / number_of_iterations_per_epoch
            running_loss = running_loss / number_of_iterations_per_epoch
            scores = None

            if self.model_is_validating:
                validation_loss, _ = self.__validate()

                if self.early_stopping(validation_loss):
                    training_end_time = time.perf_counter()
                    self.info.add_runtime(training_end_time - training_start_time)

                    self.save_model_training()

                    return None

            self.history.step(float(running_loss), running_accuracy, scores)

        print('Finished Training.')

        training_end_time = time.perf_counter()
        self.info.add_runtime(training_end_time - training_start_time)

        self.save()
        return None

    def __train_batch(self, batch_loader: callable,
                      batch_size: int=256, criterion=nn.CrossEntropyLoss()) -> (float, float):

        valid_batch_size = min(len(self.data_loader), batch_size)

        batch = batch_loader(valid_batch_size)

        inputs, masks = batch.to_tensors(self.word_vectors, pad_value=-1e-20, max_length=self.max_length)
        labels = batch.labels_encoding
        del batch

        # Put all on GPU
        inputs = inputs.to(self.hyper_parameters.device)
        masks = masks.to(self.hyper_parameters.device)
        labels = labels.to(self.hyper_parameters.device)

        # Zero the parameter gradients.
        self.optimizer.zero_grad()

        # Forward -> backward -> optimizer
        outputs = self.model(inputs, masks)
        predictions = self._minibatch_predictions(outputs)

        loss = criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy

    def __validate(self, sampling_batch_size: int=256, criterion=nn.CrossEntropyLoss()) -> (float, float):
        if not self.model_is_validating:
            raise ModelIsNotValidatingError

        validation_loss, validation_accuracy = self.test(self.__validation_data_loader,
                                                         test_batch_size=sampling_batch_size,
                                                         criterion=criterion, print_test_type='validation')

        self.validation_history.step(validation_loss, validation_accuracy)
        return validation_loss, validation_accuracy

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

    def test(self, test_data_loader, test_batch_size: int = 256,
             criterion=nn.CrossEntropyLoss(), print_test_type: str='test') -> (float, float):

        self.model.eval()
        max_batch_size = min(len(test_data_loader), test_batch_size)

        number_of_test_iterations = len(test_data_loader) // max_batch_size

        number_guessed_correctly = 0
        loss = 0
        for i in range(number_of_test_iterations):
            test_data = test_data_loader.load_clean_batch_sequential(batch_size=max_batch_size)
            lines, masks = test_data.to_tensors(self.word_vectors, pad_value=1e-20, max_length=self.max_length)
            labels = test_data.labels_encoding

            # To device (GPU)
            lines = lines.to(self.hyper_parameters.device)
            masks = masks.to(self.hyper_parameters.device)
            labels = labels.to(self.hyper_parameters.device)

            outputs = self.model(lines, masks)

            for label, prediction in zip(labels, torch.argmax(outputs, dim=1)):
                number_guessed_correctly += int(label == prediction)
                loss += criterion(outputs, labels)

        loss = float(loss)

        accuracy = number_guessed_correctly / (number_of_test_iterations * max_batch_size)
        loss = loss / (number_of_test_iterations * max_batch_size)
        print(f'Mean {print_test_type} loss: {round(loss, 4)}. '
              f'Mean {print_test_type} accuracy: {round(accuracy * 100, 2)}%')
        print('=' * 50)
        self.model.train()
        return loss, accuracy

    def save(self) -> None:
        super().save()
        if self.model_is_validating:
            self.validation_history.save()
        return None

    def save_model_training(self) -> None:
        super().save_model_training()
        if self.model_is_validating:
            self.validation_history.save()
        return None

    @staticmethod
    def __print_step(epoch, batch_step, loss, accuracy):
        print('[%d, %5d] loss: %.4f \t accuracy: %.2f' %
              (epoch + 1, batch_step + 1, float(loss), 100 * accuracy))

    @staticmethod
    def __assert_valid_batch_load_config(config: str) -> None:
        valid_configs = StaticEntailmentNet.batch_load_configs
        error_message = f"Invalid batch load config: {config}. Try one of {valid_configs}"
        assert config in valid_configs, Exception(error_message)
        return None

    def __load_clean_batch(self, batch_load_config="random"):
        self.__assert_valid_batch_load_config(batch_load_config)

        if batch_load_config == "random":
            return self.data_loader.load_clean_batch_random

        if batch_load_config == "sequential":
            return self.data_loader.load_clean_batch_sequential

        # If a batch load config is given that is somehow valid:
        #  -> it must not be implemented.
        raise NotImplementedError
