import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUNet(nn.Module):
    def __init__(self, args):
        super(GRUNet, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.encoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True
        )

        # predictor
        self.end_conv = nn.Conv1d(
            1,
            args.horizon * self.output_dim,
            kernel_size=(self.hidden_dim),
            bias=True,
        )

    def forward(self, source):
        # source: B, T_1
        source = source.unsqueeze(-1)
        output, _ = self.encoder(source)  # B, T, N, hidden
        output = output[:, -1:, :]  # B, 1, hidden

        # CNN based predictor
        output = self.end_conv(output)  # B, T_out, 1
        output = output.squeeze(-1)

        return output
