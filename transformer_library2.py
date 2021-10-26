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
        self.embed_size = 300
        self.num_layers = num_layers
        self.forward_expansion = forward_expansion
        self.heads = heads
        self.dropout = dropout
        self.device = device
        self.batch_size = batch_size


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
        self.fully_connected_out = nn.Linear(heads * self.head_dimension, embed_size)

    def forward(self, values: np.array, keys: np.array, query: np.array, mask):
        """ Multi-head concat Attention."""

        # We have a (b, seq_len, e, num_sent) shape input
        # We can ONLY pass in a (b, seq_len, head_dim) shape tensor

        n = query.shape[0]  # # Batch Size

        # Max_pad_len
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        num_sentences = values.shape[3]

        # Split pieces into self.heads pieces

        values = values.reshape(n, value_len, num_sentences, self.heads, self.head_dimension)
        keys = keys.reshape(n, key_len, num_sentences, self.heads, self.head_dimension)
        queries = query.reshape(n, query_len, num_sentences, self.heads, self.head_dimension)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Multiply the queries with the keys. Q K^T = energy
        #   queries shape: (n, query_len, num_sent, heads, heads_dim)
        #   keys shape: (n, keys_len, num_sent, heads, heads_dim)
        # energy shape: (n, heads, num_sent, query_len, key_len)

        energy = torch.einsum("nqshd,nkshd->nhsqk", [queries, keys])

        # TODO fix this problematic code
        if mask is not None:
            # Mask shape is         (n, seq_len, e, num_sent)
            # Energy shape is       (n, h, num_sent, seq_len, seq_len)
            # Desired mask shape is (n, h, num_sent, seq_len)

            # Swap the columns around to match energy shape
            mask_reshaped = mask.reshape(n, num_sentences, self.embed_size, value_len)
            # Drop the 2nd index dimension. (embed size)
            mask_reshaped = mask_reshaped[:, :, 0, :]
            # Repeat along final dim self.heads times
            # The swap the columns back around.
            mask_reshaped = mask_reshaped.unsqueeze(-1)
            mask_reshaped = mask_reshaped.repeat(1, 1, 1, self.heads)
            mask_reshaped = mask_reshaped.reshape(n, self.heads, num_sentences, value_len)
            mask_reshaped = mask_reshaped.unsqueeze(-1).repeat((1, 1, 1, 1, value_len))

            energy = energy.masked_fill(mask_reshaped == 0, float("-1e20"))

        attention_softmax = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        attention = torch.einsum("nhsql,nlshd->nqhds", [attention_softmax, values])
        # Back to the original shape
        attention = attention.reshape(n, query_len, num_sentences, self.heads * self.head_dimension)
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
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        query_reshaped = query.reshape(query.shape[0], query.shape[1], query.shape[3], query.shape[2])
        x = self.dropout(self.norm1(attention + query_reshaped))  # Skip connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))  # Skip connection
        out = out.reshape(out.shape[0], out.shape[1], out.shape[3], out.shape[2])
        return out


class EntailmentEncoder(nn.Module):
    def __init__(self, num_sentences: int, max_seq_len: int, hyper_parameters: HyperParams = HyperParams()):
        super(EntailmentEncoder, self).__init__()

        # Input shape: (batch_size, max_length, embed_size, num_sentences)
        self.num_sentences = num_sentences
        self.hyper_params = hyper_parameters
        self.max_length = max_seq_len
        self.embed_size = hyper_parameters.embed_size
        self.device = hyper_parameters.device

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
        batch_size, sequence_length, embed_size, num_sentences = x.shape
        assert num_sentences == self.num_sentences
        # Positional encoding
        positions = torch.arange(0, sequence_length).expand(batch_size, sequence_length)

        # Shape (b, seq_len, e, 2)
        positions.to(self.device)

        positions_out = self.position_embedding(positions)
        positions_out = positions_out.unsqueeze(-1).repeat((1, 1, 1, num_sentences))  # Duplicate across num sentences

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
        self.batch_size, self.max_length, self.embed_size, self.num_sentences = data_shape
        print('Batch Default Shape:', data_shape)
        self.hyper_params = hyper_parameters
        self.hyper_params.embed_size = self.embed_size

        self.encoder_flattened_size = self.max_length * self.embed_size * self.num_sentences

        self.encoder = EntailmentEncoder(self.num_sentences, max_seq_len, self.hyper_params)
        self.fc1 = nn.Linear(self.encoder_flattened_size, self.embed_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.embed_size, number_of_output_classes)

        self.device = self.hyper_params.device

    def forward(self, x, mask):
        # Input shape: (batch_size, max_length, embed_size, num_sentences)
        x = self.encoder(x, mask)

        x = x.reshape(self.batch_size, self.encoder_flattened_size)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)  # OUTPUT MUST BE POSITIVE

        return x


class EntailmentNet:
    def __init__(self, word_vectors, data_loader: SNLI_DataLoader,
                 hyper_parameters: HyperParams = HyperParams()):
        self.data_loader = data_loader
        self.word_vectors = word_vectors
        self.__hyper_parameters = hyper_parameters

        self.batch_size = hyper_parameters.batch_size
        self.embed_size = word_vectors.d_emb
        self.num_sentences = data_loader.num_sentences
        self.input_shape = (self.batch_size, data_loader.max_words_in_sentence_length, self.embed_size, self.num_sentences)

        # TODO make this auto from data loader
        self.num_classes = 3

        self.transformer = EntailmentTransformer(self.input_shape, max_seq_len=data_loader.max_words_in_sentence_length,
                                                 hyper_parameters=hyper_parameters,
                                                 number_of_output_classes=self.num_classes)

        self.optimizer = optim.SGD(self.transformer.parameters(), lr=0.001, momentum=0.9)

    @property
    def hyper_parameters(self):
        return self.__hyper_parameters

    def train(self, epochs: int, criterion=nn.CrossEntropyLoss(), print_every: int = 1):
        number_of_iterations_per_epoch = len(self.data_loader) // self.batch_size
        for epoch in range(epochs):
            for i in range(number_of_iterations_per_epoch):
                percentage_complete = round(i*100/number_of_iterations_per_epoch, 1)
                print(f'Training batch {i} of {number_of_iterations_per_epoch}. {percentage_complete}% done')
                running_loss = 0.0

                batch = self.data_loader.load_batch_random(self.batch_size).to_model_data()
                batch.clean_data()

                inputs, masks = batch.to_tensors(self.word_vectors)
                labels = batch.labels_encoding
                del batch

                # Zero the parameter gradients.
                self.optimizer.zero_grad()

                # Forward -> backward -> optimizer
                outputs = self.transformer(inputs, masks)

                # print('MODEL OUTPUT:', outputs)
                # print('LABELS:', labels)
                # print('OUTPUT SHAPE:', outputs.shape, 'LABEL SHAPE:', labels.shape)

                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % print_every == print_every - 1:  # print every p_e mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / print_every))
                    running_loss = 0.0
                print('-' * 20)

            # # Instead of sequentially loading data:
            # #   we will sample ceil(len(rows) /batch_size) iterations of random batches
            # number_of_iterations = int(np.ceil(len(self.data_loader) / batch_size))
            #
            # for i in range(number_of_iterations):

        print('Finished Training.')


def main():
    pass


if __name__ == "__main__":
    main()
