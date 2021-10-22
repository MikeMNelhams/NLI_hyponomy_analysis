import numpy as np
import torch
import torch.nn as nn

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
        n = query.shape[0]
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
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention_softmax = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

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
                 embed_size: int=256, num_layers: int=6,
                 forward_expansion: int=4, heads: int=8, dropout: float=0,
                 device='cuda', max_length: int=100
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
