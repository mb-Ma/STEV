import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import numpy as np


class DynGraphWave(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        N, D = cfg.max_nodes, cfg.node_embed
        self.node1 = nn.Parameter(
            torch.randn(N, D), requires_grad=True)
        if cfg.model_name == "dyngraphwave":
            from .cons_graphwave import GraphWave as model
        elif cfg.model_name == "dynagcrn":
            from .cons_agcrn import AGCRN as model
        else:
            raise ImportError("No model")
        self.model = model(cfg)
        nn.init.xavier_normal_(self.node1)

        if self.cfg.dynamic:
            self.day_emb = nn.Embedding(24, 10)
            self.week_emb = nn.Embedding(7, 10)

    def forward(self, batch):
        """
        batch.x shape (n, 12) # sample_num * node_num
        """
        if not self.cfg.is_pregraph:
            adj = torch.matmul(self.node1, self.node1.T)
            adj = torch.sigmoid(adj)
            sample_adj = adj.detach().abs()>0.5 # whether adding abs is unnecessary.

            # similar to previous graph structure learning
            # adj = F.softmax(F.relu(torch.mm(self.node1, self.node1.T)), dim=1)

            # another method
            # adj = torch.matmul(self.node1, self.node1.T)
            # TopK

            ptr = batch.ptr
            t_e = []
            edge_attr = []

            if self.cfg.dynamic:
                batch.time = batch.time.reshape(-1, 2)
                day = self.day_emb(batch.time[:, 0]) # (B, d)
                week = self.week_emb(batch.time[:, 1]) # (B, d)
                d_w = torch.cat([day, week],dim=1).unsqueeze(1)
                time_values = torch.einsum('bld, bdc->blc', d_w, d_w.transpose(1,2)).reshape(-1) # B 

            for i in range(len(batch.ptr) - 1):
                # commented codes are sparse version.
                # s_N = ptr[i]
                # e_N = ptr[i + 1] - ptr[i]
                # sub_adj = sparse_adj[
                #     :, (sparse_adj[0] < e_N) & (sparse_adj[1] < e_N)
                # ]
                # t_e.append(sub_adj + s_N)
                # edge_attr.append(adj[sub_adj[0], sub_adj[1]])
                s_N = ptr[i]
                e_N = ptr[i + 1] - ptr[i]
                # import pdb; pdb.set_trace()
                sub_adj = sample_adj[:e_N, :e_N]
                sub_adj = sub_adj.nonzero().T.long() # 2, n
                t_e.append(sub_adj + s_N)
                # build time-varing matrix
                # time_adj = adj[sub_adj[0], sub_adj[1]] * time_values[i]
                if self.cfg.dynamic:
                    edge_attr.append(adj[sub_adj[0], sub_adj[1]]*time_values[i])
                else:
                    edge_attr.append(adj[sub_adj[0], sub_adj[1]])

            edge_index = torch.concat(t_e, axis=1)
            edge_attr = torch.concat(edge_attr, axis=0).unsqueeze(1)
        else:
            edge_index, edge_attr = batch.edge_index, batch.edge_attr
        
        if self.cfg.is_cons_loss:
            out, feat = self.model(batch.x.unsqueeze(2), edge_index, edge_attr)
            return out.squeeze(2), feat
        else:
            out = self.model(batch.x.unsqueeze(2), edge_index, edge_attr)
            return out.squeeze(2)
        
