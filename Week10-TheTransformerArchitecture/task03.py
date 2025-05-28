import torch
import torch.nn as nn
import math

VOCAB_SIZE = 10000
D_MODEL = 512
MAX_SEQ_LENGTH = 4

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


def main():
    embedding_layer = InputEmbeddings(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    pos_encoder = PositionalEncoding(d_model=D_MODEL,
                                     max_seq_length=MAX_SEQ_LENGTH)

    token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    embedded = embedding_layer(token_ids)
    output = pos_encoder(embedded)

    print("Shape of output:", output.shape)
    print("Encoding of first token embedding:", output[0, 0])


if __name__ == '__main__':
    main()
