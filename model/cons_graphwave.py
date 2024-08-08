import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, ChebConv, Sequential  # GPSConv,
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class gnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.gcns = ChebConv(in_channels, out_channels, K=2)

    def forward(self, x, edge_index, edge_attr):
        return self.gcns(x, edge_index, edge_attr)


class NLayer(nn.Module):
    def __init__(
        self,
        dilation: int = 1,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        kernel_size: int = 2,
        seq_len: int = 12,
    ):
        super().__init__()

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.kernel_size = kernel_size
        self.skip_channels = skip_channels

        self.dilation = dilation

        self.filter_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.residual_channels,
                out_channels=self.dilation_channels,
                kernel_size=(1, self.kernel_size),
                dilation=self.dilation,
            ),
            nn.Tanh(),
        )

        self.gate_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.residual_channels,
                out_channels=self.dilation_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
            ),
            nn.Sigmoid(),
        )

        self.residual_conv = nn.Conv1d(
            in_channels=self.dilation_channels,
            out_channels=self.residual_channels,
            kernel_size=1,
        )

        self.skip_conv = nn.Conv1d(
            in_channels=self.dilation_channels,
            out_channels=self.skip_channels,
            kernel_size=1,
        )

        self.bn = nn.BatchNorm2d(self.residual_channels)

    def forward(self, x, skip):
        """

        Args:
            x (torch.Tensor): [N, C, F, T]

        Returns:
            x: torch.Tensor: [N, C, F, T]
            skip: torch.Tensor: [N, F, T]
        """
        residual = x
        filters = self.filter_conv(residual)
        gate = self.gate_conv(residual.squeeze(2))

        x = filters * gate.unsqueeze(2)

        s = self.skip_conv(x.squeeze(2))
        if skip is not 0:
            skip = skip[:, :, -s.size(2) :]
        skip = s + skip

        x = x + residual[:, :, :, -x.size(3) :]
        x = self.bn(x)
        return x, skip


class GraphWave(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.input_size
        self.output_dim = cfg.output_size
        self.horizon = cfg.horizon
        self.use_norm = cfg.use_norm
        self.is_cons_loss = cfg.is_cons_loss
        self.num_layers = cfg.num_layers
        self.num_blocks = 4
        kernel_size = 2
        residual_channels = 32
        dilation_channels = 32
        skip_channels = 256
        end_channels = 512

        self.start_conv = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        receptive_field = 1
        seq_len = 13
        self.layers = nn.ModuleList()
        self.gcns = nn.ModuleList()
        t_count = 0
        for b in range(self.num_blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(self.num_layers):
                seq_len -= new_dilation
                self.layers.append(
                    NLayer(
                        dilation=new_dilation,
                        residual_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        seq_len=seq_len,
                    )
                )
                if (t_count) % 4 == 0:
                    self.gcns.append(
                        gnnBlock(
                            in_channels=dilation_channels * seq_len,
                            out_channels=dilation_channels  * seq_len,
                            hidden_channels=32,
                        )
                    )
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                t_count += 1

        self.end_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=skip_channels,
                out_channels=end_channels,
                kernel_size=(1, 1),
                bias=True,
            ),
            nn.Conv2d(
                in_channels=end_channels,
                out_channels=self.horizon,
                kernel_size=(1, 1),
                bias=True,
            ),
        )
        self.receptive_field = receptive_field

        if cfg.is_cons_loss:
            self.project = nn.Sequential(
                nn.Linear(skip_channels, skip_channels),
                nn.BatchNorm1d(skip_channels),
                nn.ReLU(),
                nn.Linear(skip_channels, skip_channels // 2)
            )

    def _pad(self, x):
        in_len = x.size(3)
        if in_len < self.receptive_field:
            return F.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        return x

    def forward(self, x, edge_index, edge_attr):
        '''
        X: [B*N, T, F=1]
        edge_attr: [node, 1]
        '''
        if self.use_norm:
            # Normalization from Non-stationary Transformer [B, L, N]
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        x = x.unsqueeze(2).permute(0, 2, 3, 1)  # [B*N, 1, F, T]
        x = self._pad(x)
        x = self.start_conv(x) # (B*N, F_hidden, 1, 13)
        skip = 0
        for i in range(self.num_blocks * self.num_layers):
            if i % 4 - 1 == 0:
                bs = x.shape
                x = x.view(bs[0], -1)
                import pdb; pdb.set_trace
                x = self.gcns[i // 4](x, edge_index, edge_attr)
                x = x.view(bs[0], bs[1], bs[2], bs[3])
            x, skip = self.layers[i](x, skip)
        
        
        x = F.relu(skip.unsqueeze(2))
        x = self.end_conv(x)
        x = x.squeeze(2) # [B, H，1]
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            x = x * (stdev.repeat(1, self.horizon, 1))
            x = x + (means.repeat(1, self.horizon, 1))
        # torch.mean(skip, dim=1).squeeze().detach().cpu().numpy()
        if self.is_cons_loss:
            feat = self.project(skip.squeeze())
            return x, torch.mean(skip, dim=1).squeeze().detach().cpu().numpy()
        else:
            return x
