import torch
import torch.nn as nn
from torch.nn import functional
import math

VOCAB_SIZE = 10000
D_MODEL = 512
MAX_SEQ_LENGTH = 20
NUM_HEADS = 8
D_FF = 2048
DROPOUT_RATE = 0.1
NUM_LAYERS = 6

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


def create_tgt_mask(token_ids):
    seq_len = token_ids.size(1)
    mask = torch.ones(seq_len, seq_len)
    mask = torch.tril(mask)

    return mask


class Head(nn.Module):

    def __init__(self, d_model, head_size):
        super().__init__()

        self.q_linear = nn.Linear(d_model, head_size, bias=False)
        self.k_linear = nn.Linear(d_model, head_size, bias=False)
        self.v_linear = nn.Linear(d_model, head_size, bias=False)

        self.d_model = d_model
        self.head_size = head_size

    def forward(self, q, k, v, mask):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

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

    def forward(self, q, k, v, mask):
        head_outputs = [head(q, k, v, mask) for head in self.heads]
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
        attention_out = self.self_attention(x, x, x, mask)
        attention_out = self.dropout(attention_out)
        x = self.norm1(x + attention_out)

        ff_out = self.ff(x)
        ff_out = self.dropout(ff_out)
        x = self.norm2(x + ff_out)

        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForwardSubLayer(d_model, d_ff)

        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, y, tgt_mask, src_mask):
        x = self.norm1(x +
                       self.dropout(self.self_attention(x, x, x, tgt_mask)))
        x = self.norm2(x +
                       self.dropout(self.cross_attention(x, y, y, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))

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

    def forward(self, x, src_mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size, d_model, max_seq_length, num_layers,
                 num_heads, d_ff, dropout_rate):
        super().__init__()

        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, x, y, tgt_mask, src_mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, y, tgt_mask, src_mask)

        return x


class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff,
                 max_seq_len, dropout_rate):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size=vocab_size,
                                          d_model=d_model,
                                          num_heads=num_heads,
                                          num_layers=num_layers,
                                          d_ff=d_ff,
                                          dropout_rate=dropout_rate,
                                          max_seq_length=max_seq_len)
        self.decoder = TransformerDecoder(vocab_size=vocab_size,
                                          d_model=d_model,
                                          num_heads=num_heads,
                                          num_layers=num_layers,
                                          d_ff=d_ff,
                                          dropout_rate=dropout_rate,
                                          max_seq_length=max_seq_len)

    def forward(self, x, src_mask, tgt_mask):
        encoder_output = self.encoder(x, src_mask)
        decoder_output = self.decoder(x, encoder_output, tgt_mask, src_mask)
        return decoder_output


class ClassifierHead(nn.Module):

    def __init__(self, d_model, num_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.log_softmax(self.linear(x))


def main():
    input_ids = torch.tensor(
        [[6044, 8239, 4933, 3760, 8963, 8379, 5427, 8503, 3497, 5683],
         [4101, 6866, 2756, 1399, 5878, 376, 56, 9868, 8794, 6033]])

    transformer = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF,
                              MAX_SEQ_LENGTH, DROPOUT_RATE)
    classifier = ClassifierHead(D_MODEL, VOCAB_SIZE)

    src_mask = create_src_mask(input_ids)
    tgt_mask = create_tgt_mask(input_ids)
    transformet_output = transformer(input_ids, src_mask, tgt_mask)
    class_logits = classifier(transformet_output)

    print(class_logits)
    print(class_logits.shape)


if __name__ == '__main__':
    main()
