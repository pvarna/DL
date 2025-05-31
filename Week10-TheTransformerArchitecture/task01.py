import torch.nn as nn


def main():
    model = nn.Transformer(d_model=1536,
                           nhead=8,
                           num_encoder_layers=6,
                           num_decoder_layers=6)

    print(model)


if __name__ == '__main__':
    main()
