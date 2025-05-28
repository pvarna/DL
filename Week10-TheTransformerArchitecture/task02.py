import torch
import torch.nn as nn

VOCAB_SIZE = 10000
EMBEDDING_DIM = 512

torch.manual_seed(123)


class InputEmbeddings(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x) * (self.embedding.embedding_dim**0.5)


def main():
    model = InputEmbeddings(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)

    token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    output = model(token_ids)

    print(f"Shape of output: {output.shape}")
    print(f"Embedding of first token: {output[0, 0]}")


if __name__ == '__main__':
    main()
