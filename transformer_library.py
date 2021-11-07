import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from SNLI_data_handling import SNLI_DataLoader


# CODE FROM: https://www.youtube.com/watch?v=U0s0f995w14&t=2494s
#   Paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf


class SelfAttention(nn.Module):
    """ Implementation Code from: https://www.youtube.com/watch?v=U0s0f995w14&t=2494s,
        From paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"""

    def __init__(self, embed_size: int, heads: int):
        super(SelfAttention, self).__init__()
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
        # Batch Size
        n = query.shape[0]

        # Max_pad_len
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(n, value_len, self.heads, self.head_dimension)
        keys = keys.reshape(n, key_len, self.heads, self.head_dimension)
        queries = query.reshape(n, query_len, self.heads, self.head_dimension)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Multiply the queries with the keys. Q K^T = energy
        #   queries shape: (n, query_len, heads, heads_dim)
        #   keys shape: (n, keys_len, heads, heads_dim)
        # energy shape: (n, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            # print('MASK:', mask.shape)
            # print('ENERGY:', energy.shape)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention_softmax = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        attention = torch.einsum("nhql,nlhd->nqhd", [attention_softmax, values])
        attention = attention.reshape(n, query_len, self.heads * self.head_dimension)
        # Full equation: attention(Q, K, V) = softmax((Q K^T)/sqrt(d_k)) V
        #   attention_softmax shape: (n, heads, query_len, key_len)
        #   values shape: (n, value_len, heads, heads_dim)
        # attention shape: (n, query_len, heads, heads_dim) then flatten last two dimensions

        attention_out = self.fully_connected_out(attention)
        return attention_out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
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

        x = self.dropout(self.norm1(attention + query))  # Skip connection
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))  # Skip connection
        print('PASSED SINGLE FF')
        return out


class Encoder(nn.Module):
    def __init__(self, source_vocab_size: int, embed_size: int, num_layers: int, heads: int,
                 device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        n, sequence_length = x.shape
        # Positional encoding
        positions = torch.arange(0, sequence_length).expand(n, sequence_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads,
                                                  dropout=dropout, forward_expansion=forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, source_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, source_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, target_vocab_size: int, embed_size: int, num_layers: int, heads: int,
                 forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, dropout=dropout,
                          forward_expansion=forward_expansion, device=device) for _ in range(num_layers)]
        )
        self.fully_connected_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, source_mask, target_mask):
        n, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).expand(n, sequence_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, source_mask, target_mask)

        out = self.fully_connected_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self, source_vocab_size: int, target_vocab_size: int,
                 source_pad_index: int, target_pad_index: int,
                 embed_size: int = 256, num_layers: int = 6,
                 forward_expansion: int = 4, heads: int = 8, dropout: float = 0,
                 device='cuda', max_length: int = 100
                 ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(source_vocab_size, embed_size, num_layers, heads,
                               device, forward_expansion, dropout, max_length)

        self.decoder = Decoder(target_vocab_size, embed_size, num_layers, heads,
                               forward_expansion, dropout, device, max_length)

        self.source_pad_index = source_pad_index
        self.target_pad_index = target_pad_index
        self.device = device

    def source_mask(self, source):
        source_mask = (source != self.source_pad_index).unsqueeze(1).unsqueeze(2)
        # (n, 1, 1, source_len)
        return source_mask.to(self.device)

    def target_mask(self, target):
        n, target_len = target.shape
        # Triangular matrix |_\
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            n, 1, target_len, target_len
        )
        return target_mask.to(self.device)

    def forward(self, source, target):
        source_mask = self.source_mask(source)
        target_mask = self.target_mask(target)

        encoder_source = self.encoder(source, source_mask)
        out = self.decoder(target, encoder_source, source_mask, target_mask)
        return out


class HyperParams:
    def __init__(self, num_layers: int = 6, forward_expansion: int = 4, heads: int = 8, dropout: float = 0,
                 device='cuda', max_sentence_len: int = 100):
        self.embed_size = 300
        self.num_layers = num_layers
        self.forward_expansion = forward_expansion
        self.heads = heads
        self.dropout = dropout
        self.device = device
        self.max_sentence_len = max_sentence_len


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
        print('Attention SHAPE:', attention.shape)
        # Full equation: attention(Q, K, V) = softmax((Q K^T)/sqrt(d_k)) V
        #   attention_softmax shape: (n, heads, query_len, key_len)
        #   values shape: (n, value_len, heads, heads_dim)
        # attention shape: (n, query_len, heads, heads_dim) then flatten last two dimensions

        attention_out = self.fully_connected_out(attention)
        return attention_out


class EntailmentTransformerBlock(TransformerBlock):
    def __init__(self, embed_size: int, heads: int, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
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
    def __init__(self, num_sentences: int, hyper_parameters: HyperParams = HyperParams()):
        super(EntailmentEncoder, self).__init__()

        # Input shape: (batch_size, max_length, embed_size, num_sentences)
        self.num_sentences = num_sentences
        self.hyper_params = hyper_parameters
        self.max_length = hyper_parameters.max_sentence_len
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
    def __init__(self, data_shape, output_shape: int = 3,
                 hyper_parameters: HyperParams = HyperParams()):
        super(EntailmentTransformer, self).__init__()

        # Input shape: (batch_size, max_length, embed_size, num_sentences)
        self.batch_size, self.max_length, self.embed_size, self.num_sentences = data_shape
        print('DATA Shape:', data_shape)
        self.hyper_params = hyper_parameters
        self.hyper_params.embed_size = self.embed_size

        self.encoder = EntailmentEncoder(self.num_sentences, self.hyper_params)
        self.encoder_flattened_size = self.batch_size * self.max_length * self.embed_size * self.num_sentences
        print(f'({self.encoder_flattened_size},{self.embed_size})')
        self.fc1 = nn.Linear(self.encoder_flattened_size, self.embed_size)
        self.fc2 = nn.Linear(self.embed_size, output_shape)

        self.device = self.hyper_params.device

    def forward(self, x, mask):
        # Input shape: (batch_size, max_length, embed_size, num_sentences)
        x = self.encoder(x, mask)

        x = torch.flatten(x)
        print(x.shape)
        x = self.fc1(x)
        x = torch.nn.ReLU(x)
        x = self.fc2(x)

        return x


class EntailmentNet:
    def __init__(self, word_vectors, data_loader: SNLI_DataLoader,
                 hyper_parameters: HyperParams = HyperParams()):
        self.data_loader = data_loader
        self.word_vectors = word_vectors
        self.__hyper_parameters = hyper_parameters

        self.embed_size = word_vectors.d_emb
        self.num_sentences = data_loader.num_sentences
        self.input_shape = (1, hyper_parameters.max_sentence_len, self.embed_size, self.num_sentences)

        # TODO make this auto from data loader
        self.num_classes = 3

        self.transformer = EntailmentTransformer(self.input_shape,
                                                 hyper_parameters=hyper_parameters, output_shape=self.num_classes)

        self.optimizer = optim.SGD(self.transformer.parameters(), lr=0.001, momentum=0.9)

    @property
    def hyper_parameters(self):
        return self.__hyper_parameters

    def train(self, epochs: int, batch_size: int=256, criterion=nn.CrossEntropyLoss(), print_every: int = 10):
        for epoch in range(epochs):
            for i in range(len(self.data_loader)):
                running_loss = 0.0

                inputs = self.data_loader.load_batch_random(1).to_model_data()
                inputs.clean_data()

                labels = inputs.labels_encoding
                inputs, masks = inputs.to_tensors(self.word_vectors)

                # Zero the parameter gradients.
                self.optimizer.zero_grad()

                # Forward -> backward -> optimizer
                outputs = self.transformer(inputs, masks)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % print_every == print_every - 1:  # print every p_e mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / print_every))
                    running_loss = 0.0

            # # Instead of sequentially loading data:
            # #   we will sample ceil(len(rows) /batch_size) iterations of random batches
            # number_of_iterations = int(np.ceil(len(self.data_loader) / batch_size))
            #
            # for i in range(number_of_iterations):


        print('Finished Training.')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    target = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    source_pad_index = 0
    target_pad_index = 0
    source_vocab_size = 10
    target_vocab_size = 10

    model = Transformer(source_vocab_size, target_vocab_size, source_pad_index, target_pad_index).to(device)
    out = model(x, target[:, :-1])
    print(out.shape)


if __name__ == "__main__":
    main()
