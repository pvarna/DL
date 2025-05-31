import torch
import torch.nn as nn
from torch.nn import functional
import math

VOCAB_SIZE = 10000
D_MODEL = 512
MAX_SEQ_LENGTH = 4
HEAD_SIZE = 512

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

        batch_size = q.size(0)
        print(
            f"Attention scores of the last batch: {attention_scores[batch_size-1]}"
        )

        attention_weights = functional.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_weights, v)

        return out


def main():
    embedding_layer = InputEmbeddings(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    pos_encoder = PositionalEncoding(d_model=D_MODEL,
                                     max_seq_length=MAX_SEQ_LENGTH)

    token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 0, 0]])

    embedded = embedding_layer(token_ids)
    pos_encoder_output = pos_encoder(embedded)

    head = Head(d_model=D_MODEL, head_size=HEAD_SIZE)

    print(f"Input shape to \"creare_src_mask\": {token_ids.shape}")
    mask = create_src_mask(token_ids)
    print(f"Output shape of \"creare_src_mask\": {mask.shape}")
    print(f"Mask: {mask}")

    head_output = head(pos_encoder_output, mask)

    print(f"Shape of output of head: {head_output.shape}")


if __name__ == '__main__':
    main()
