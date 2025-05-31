import torch
import torch.nn as nn
from torch.nn import functional
import math

VOCAB_SIZE = 10000
D_MODEL = 512
MAX_SEQ_LENGTH = 4
NUM_HEADS = 4
D_FF = 2048
DROPOUT_RATE = 0.1
NUM_LAYERS = 4

torch.manual_seed(123)


class InputEmbeddings(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * (self.embedding.embedding_dim**0.5)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def create_src_mask(token_ids):
    mask = torch.ones_like(token_ids)
    mask[token_ids == 0] = 0
    mask = mask.unsqueeze(1)
    mask = mask.expand(-1, token_ids.size(1), -1)

    return mask


class Head(nn.Module):

    def __init__(self, d_model, head_size):
        super().__init__()

        self.q_linear = nn.Linear(d_model, head_size, bias=False)
        self.k_linear = nn.Linear(d_model, head_size, bias=False)
        self.v_linear = nn.Linear(d_model, head_size, bias=False)

        self.d_model = d_model
        self.head_size = head_size

    def forward(self, x, mask):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        k_T = torch.transpose(k, -2, -1)

        scale = self.head_size**0.5
        attention_scores = torch.matmul(q, k_T) / scale

        attention_scores = attention_scores.masked_fill(mask == 0, -torch.inf)

        attention_weights = functional.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_weights, v)

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(d_model, d_model // num_heads) for _ in range(num_heads)])
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        head_outputs = [head(x, mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)

        return self.output_linear(concatenated)


class FeedForwardSubLayer(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.ff(x)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = FeedForwardSubLayer(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attention_out = self.self_attention(x, mask)
        attention_out = self.dropout(attention_out)
        x = self.norm1(x + attention_out)

        ff_out = self.ff(x)
        ff_out = self.dropout(ff_out)
        x = self.norm2(x + ff_out)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size, d_model, max_seq_length, num_layers,
                 num_heads, d_ff, dropout_rate):
        super().__init__()

        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        mask = create_src_mask(x)
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)

        return x


def main():
    token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 0, 0]])

    print(f"Input shape: {token_ids.shape}")

    encoder = TransformerEncoder(vocab_size=VOCAB_SIZE,
                                 d_model=D_MODEL,
                                 max_seq_length=MAX_SEQ_LENGTH,
                                 num_layers=NUM_LAYERS,
                                 num_heads=NUM_HEADS,
                                 d_ff=D_FF,
                                 dropout_rate=DROPOUT_RATE)

    output = encoder(token_ids)
    print(f"Output shape: {output.shape}")
    print(f"First element in first batch: {output[0, 0]}")


if __name__ == '__main__':
    main()
