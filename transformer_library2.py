import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from SNLI_data_handling import SNLI_DataLoader


# CODE FROM: https://www.youtube.com/watch?v=U0s0f995w14&t=2494s
#   Paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf


class HyperParams:
    def __init__(self, num_layers: int = 6, forward_expansion: int = 4, heads: int = 8, dropout: float = 0,
                 device='cuda', batch_size: int = 256):
        self.num_layers = num_layers
        self.forward_expansion = forward_expansion
        self.heads = heads
        self.dropout = dropout
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EntailmentSelfAttention(nn.Module):
    """ Implementation Code MODIFIED from: https://www.youtube.com/watch?v=U0s0f995w14&t=2494s,
            From paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"""

    def __init__(self, embed_size: int, heads: int):
        super(EntailmentSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dimension = embed_size // heads

        assert (self.head_dimension * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.keys = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.queries = nn.Linear(self.head_dimension, self.head_dimension, bias=False)
        self.fully_connected_out = nn.Linear(embed_size, embed_size)

    def forward(self, values: np.array, keys: np.array, query: np.array, mask):
        """ Multi-head concat Attention."""

        n = query.shape[0]  # Batch Size
        value_len, key_len, query_len = values.shape[2], keys.shape[2], query.shape[2]  # Max sequence len
        num_sentences = values.shape[1]

        # Split pieces into self.heads pieces
        values = values.reshape(n, num_sentences, value_len, self.heads, self.head_dimension)
        keys = keys.reshape(n, num_sentences, key_len, self.heads, self.head_dimension)
        queries = query.reshape(n, num_sentences, query_len, self.heads, self.head_dimension)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

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
            # TODO Rigorously PROVE this is correct!
            mask_reshaped = mask_reshaped.unsqueeze(-1).repeat(1, 1, 1, self.heads)
            mask_reshaped = mask_reshaped.permute(0, 1, 3, 2)
            mask_reshaped = mask_reshaped.unsqueeze(-1).repeat((1, 1, 1, 1, value_len))

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
    def __init__(self, embed_size: int, heads: int, dropout, forward_expansion):
        super(EntailmentTransformerBlock, self).__init__()
        self.attention = EntailmentSelfAttention(embed_size=embed_size, heads=heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.Sigmoid(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))  # Skip connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))  # Skip connection
        return out


class EntailmentEncoder(nn.Module):
    def __init__(self, num_sentences: int, max_seq_len: int, embed_size: int=300,
                 hyper_parameters: HyperParams = HyperParams()):
        super(EntailmentEncoder, self).__init__()

        # Input shape: (batch_size, max_length, embed_size, num_sentences)
        self.num_sentences = num_sentences
        self.hyper_params = hyper_parameters
        self.max_length = max_seq_len
        self.embed_size = embed_size

        self.position_embedding = nn.Embedding(self.max_length, self.embed_size)

        self.layers = nn.ModuleList(
            [
                EntailmentTransformerBlock(self.embed_size,
                                           self.hyper_params.heads,
                                           dropout=self.hyper_params.dropout,
                                           forward_expansion=self.hyper_params.forward_expansion)
                for _ in range(self.hyper_params.num_layers)
            ]
        )

        self.dropout = nn.Dropout(self.hyper_params.dropout)

    def forward(self, x, mask):
        batch_size, num_sentences, sequence_length, embed_size = x.shape
        assert num_sentences == self.num_sentences
        # Positional encoding
        positions = torch.arange(0, sequence_length).expand(batch_size, sequence_length)

        # Shape (b, 2, seq_len, e)
        positions_out = self.position_embedding(positions)

        positions_out = positions_out.unsqueeze(-1).repeat((1, 1, 1, num_sentences))  # Duplicate across num sentences
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

        # Input shape: (batch_size, max_length, embed_size, num_sentences)
        self.batch_size, self.num_sentences, self.max_length, self.embed_size = data_shape
        print('Batch Default Shape:', data_shape)
        self.hyper_params = hyper_parameters
        self.hyper_params.embed_size = self.embed_size

        self.encoder_flattened_size = self.num_sentences * max_seq_len * self.embed_size

        self.encoder = EntailmentEncoder(self.num_sentences, max_seq_len,
                                         embed_size=self.embed_size, hyper_parameters=self.hyper_params)

        self.fc1 = nn.Linear(self.encoder_flattened_size, self.embed_size * max_seq_len)

        self.fc2 = nn.Linear(self.embed_size * max_seq_len, number_of_output_classes, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        # Input shape: (batch_size, num_sentences, embed_size, max_length)
        x = self.encoder(x, mask)

        x = x.reshape(self.batch_size, self.encoder_flattened_size)

        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)

        return x


class EntailmentNet:
    def __init__(self, word_vectors, data_loader: SNLI_DataLoader,
                 hyper_parameters: HyperParams = HyperParams()):
        self.data_loader = data_loader
        self.word_vectors = word_vectors
        self.__hyper_parameters = hyper_parameters

        self.device = hyper_parameters.device

        self.batch_size = hyper_parameters.batch_size
        self.embed_size = word_vectors.d_emb
        self.num_sentences = data_loader.num_sentences
        self.input_shape = (self.batch_size, self.num_sentences,
                            data_loader.max_words_in_sentence_length, self.embed_size)

        # TODO make this auto from data loader
        self.num_classes = 3

        self.transformer = EntailmentTransformer(self.input_shape, max_seq_len=data_loader.max_words_in_sentence_length,
                                                 hyper_parameters=hyper_parameters,
                                                 number_of_output_classes=self.num_classes)
        self.optimizer = optim.Adadelta(self.transformer.parameters())
        # self.optimizer = optim.SGD(self.transformer.parameters(), lr=0.001, momentum=0.9)

    @property
    def hyper_parameters(self):
        return self.__hyper_parameters

    @staticmethod
    def permute_inputs(inputs: torch.Tensor):
        return inputs.permute(0, 3, 1, 2)

    def train(self, epochs: int, criterion=nn.CrossEntropyLoss(), print_every: int = 1):
        number_of_iterations_per_epoch = len(self.data_loader) // self.batch_size
        for epoch in range(epochs):
            running_loss = 0.0
            for i in range(number_of_iterations_per_epoch):
                percentage_complete = round(i*100/number_of_iterations_per_epoch, 1)
                print(f'Training batch {i} of {number_of_iterations_per_epoch}. {percentage_complete}% done')

                batch = self.data_loader.load_batch_sequential(self.batch_size).to_model_data()
                batch.clean_data()

                inputs, masks = batch.to_tensors(self.word_vectors)
                inputs, masks = self.permute_inputs(inputs), self.permute_inputs(masks)

                labels = batch.labels_encoding
                del batch

                # Zero the parameter gradients.
                self.optimizer.zero_grad()

                # Forward -> backward -> optimizer
                outputs = self.transformer(inputs, masks)
                predictions = self.minibatch_predictions(outputs)
                print('MODEL OUTPUT:\n-----------------')
                print(predictions)
                print('LABELS:\n-----------------')
                print(labels)

                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % print_every == print_every - 1:  # print every p_e mini-batches
                    print('[%d, %5d] loss: %.3f \t accuracy: %.2f' %
                          (epoch + 1, i + 1, running_loss / print_every, 100 * self.accuracy(predictions, labels)))
                    running_loss = 0.0
                print('-' * 20)

        print('Finished Training.')
        print('Saving model...')
        torch.save(self.transformer, 'data/BERT-MIKE-MODEL0/lstmmodelgpu.pth')

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
    def minibatch_predictions(x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(x, dim=1)

    @staticmethod
    def accuracy(x: torch.Tensor, y: torch.Tensor):
        correct = 0
        for x_row, y_row in zip(x, y):
            correct += int(x_row == y_row)
        accuracy = correct / int(x.shape[0])
        return accuracy


def main():
    pass


if __name__ == "__main__":
    main()
