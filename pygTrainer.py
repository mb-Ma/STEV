import torch
import torch.nn.functional as F
import numpy as np


from baseTrainer import baseTrainer


class Trainer(baseTrainer):
    def __init__(self, model, scaler, cfg):
        super().__init__(model, scaler, cfg)
        self.cfg = cfg

    def forward(self, batch):
        batch = batch.to(self.device)
        y = batch.y
        out = self.model(batch)
        return out, y, batch.n_id


class CosTrainer(baseTrainer):
    def __init__(self, model, scaler, cfg):
        super().__init__(model, scaler, cfg)
        self.cfg = cfg

    def cal_cons_loss(self, anchor, pos, focal_weights, temperature, ptr):
        """
        (b*n, c) * (c, b*n) denotes the similarity between nodes in all samples.
        """
        anchor = F.normalize(anchor, dim=1)
        pos = F.normalize(pos, dim=1)

        # filter weak negative, same subgraph as zeros, pos_sim = 0
        weak_neg_mask = torch.ones(ptr[-1], ptr[-1]).to(anchor.device)
        for i in range(len(ptr)-1):
            weak_neg_mask[ptr[i]:(ptr[i+1]-1), ptr[i]:(ptr[i+1]-1)] = 0

        # # 最初的极简版，存在loss为负的问题
        sim_matrix = torch.matmul(anchor, pos.T) / temperature # (bs*n, bs*n)
        sim_matrix = torch.exp(sim_matrix)
        pos_sim = torch.diag(sim_matrix) # bs*n
        neg_sim = torch.sum(sim_matrix*weak_neg_mask, dim=1) + pos_sim
        # neg_sim = torch.sum(sim_matrix, dim=1)

        cons_loss = pos_sim / neg_sim * focal_weights
        # cons_loss = pos_sim / neg_sim
        cons_loss = - torch.log(cons_loss).mean()


        # # consider negative filter, do masking
        # # sim_matrix_masked = sim_matrix.masked_fill(~(weak_neg_mask.bool()), float('-inf'))
        # prob = F.softmax(sim_matrix, dim=1)
        # pos = torch.diag(prob).unsqueeze(1) # (bs*n, 1)
        # # mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool).to(anchor.device)
        # # neg_sum = torch.sum(prob[~mask].view(prob.size(0), -1), dim=1) # (bs*n, 1) remove positive pair
        # neg_sum = torch.sum(prob*weak_neg_mask, dim=1) # (bs*n, 1) remove positive pair
        # # neg_sum = torch.sum(prob, dim=1) # (bs*n, 1)
        # # reduce the weight of expanding variables
        # # cons_loss = F.softplus(torch.mean(-torch.log(pos / neg_sum)))
        # # cons_loss = torch.mean(-torch.log(pos / neg_sum * focal_weights.to(anchor.device)))
        # cons_loss = torch.mean(-torch.log(pos * focal_weights.to(anchor.device)))

        # # filter weak negative, same subgraph as zeros
        # weak_neg_mask = torch.ones(ptr[-1], ptr[-1]).to(anchor.device)
        # for i in range(len(ptr)-1):
        #     weak_neg_mask[ptr[i]:(ptr[i+1]-1), ptr[i]:(ptr[i+1]-1)] = 0

        # # 最初的极简版，存在loss为负的问题
        # sim_matrix = torch.matmul(anchor, pos.T) / temperature # (bs*n, bs*n)
        # # consider negative filter, do masking
        # # sim_matrix_masked = sim_matrix.masked_fill(~(weak_neg_mask.bool()), float('-inf'))
        # prob = F.softmax(sim_matrix, dim=1)
        # pos = torch.diag(prob).unsqueeze(1) # (bs*n, 1)
        # # mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool).to(anchor.device)
        # # neg_sum = torch.sum(prob[~mask].view(prob.size(0), -1), dim=1) # (bs*n, 1) remove positive pair
        # neg_sum = torch.sum(prob*weak_neg_mask, dim=1) # (bs*n, 1) remove positive pair
        # # neg_sum = torch.sum(prob, dim=1) # (bs*n, 1)
        # # reduce the weight of expanding variables
        # # cons_loss = F.softplus(torch.mean(-torch.log(pos / neg_sum)))
        # # cons_loss = torch.mean(-torch.log(pos / neg_sum * focal_weights.to(anchor.device)))
        # cons_loss = torch.mean(-torch.log(pos * focal_weights.to(anchor.device)))


        # # 最初的极简版，存在loss为负的问题，过早用softmax计算，计算概率
        # sim_matrix = torch.matmul(anchor, pos.T) / temperature # (bs*n, bs*n)
        # # consider negative filter, do masking
        # # sim_matrix_masked = sim_matrix.masked_fill(~(weak_neg_mask.bool()), float('-inf'))
        # prob = F.softmax(sim_matrix, dim=1)
        # pos = torch.diag(prob).unsqueeze(1) # (bs*n, 1)
        # # mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool).to(anchor.device)
        # # neg_sum = torch.sum(prob[~mask].view(prob.size(0), -1), dim=1) # (bs*n, 1) remove positive pair
        # neg_sum = torch.sum(prob*weak_neg_mask, dim=1) # (bs*n, 1) remove positive pair
        # # neg_sum = torch.sum(prob, dim=1) # (bs*n, 1)
        # # reduce the weight of expanding variables
        # # cons_loss = F.softplus(torch.mean(-torch.log(pos / neg_sum)))
        # # cons_loss = torch.mean(-torch.log(pos / neg_sum * focal_weights.to(anchor.device)))
        # cons_loss = torch.mean(-torch.log(pos * focal_weights.to(anchor.device)))
        

        #--------------------------------------------------------------------------------------------------
        # sim_matrix = torch.matmul(anchor, pos.T)  # (bs*n, bs*n)
        # sim_matrix = torch.exp(sim_matrix / 0.1) # temperature 0.1
        # mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool).to(
        #     anchor.device
        # )
        # pos_sum = torch.sum(sim_matrix * mask, dim=1)  # (bs*n, 1)
        # neg_sum = torch.sum(
        #     sim_matrix, dim=1
        # )  # (bs*n, 1)
        
        # cons_loss = torch.mean(
        #     -torch.log((pos_sum.to(anchor.device) * focal_weights.to(anchor.device))/ neg_sum.to(anchor.device))
        # )

        # InfoNCE loss 0.8796 
        # N = len(anchor)
        # logits = (torch.matmul(anchor, pos.T)) / temperature
        # mask = (~torch.eye(N, dtype=bool, device=anchor.device))
        # negative_samples = logits[mask].view(N, -1)
        # pos_samples = (torch.diag(logits)).unsqueeze(1) 
        # logits = torch.cat((pos_samples, negative_samples), dim=1)
        # labels = torch.arange(N, device=anchor.device).long() # N classification task
        # cons_loss = F.cross_entropy(logits, labels, reduction="none")
        # cons_loss = cons_loss.mean()

        # DCL 0.8445  也存在为负数的情况
        # logits = (torch.matmul(anchor, pos.T))
        # positive_loss = -torch.diag(logits) / temperature * focal_weights
        # neg_similarity = torch.cat((torch.matmul(anchor, anchor.T), logits), dim=1) / temperature # [B, 2B]
        # neg_mask = torch.eye(anchor.size(0), device=anchor.device).repeat(1, 2) # [B, 2B]
        # negative_loss = torch.logsumexp(neg_similarity + neg_mask * np.log(1e-45), dim=1, keepdim=False)
        # cons_loss = (positive_loss + negative_loss).mean()
        
        return cons_loss

    def forward(self, batch):
        # constrcut an index to denotes which sample is expanding variable.
        # if weight < 1.0, the expanding variables gain more attention. 
        focal_weights = torch.where(batch.n_id < self.cfg.first_feats, 1.0, self.cfg.weight).to(self.device)

        batch = batch.to(self.device)
        out, feat = self.model(batch)
        batch.x = batch.aug_x
        aug_out, feat_pos = self.model(batch)
        
        cons_loss = self.cal_cons_loss(feat, feat_pos, focal_weights, self.cfg.temperature, batch.ptr)

        return out, batch.y, batch.n_id, cons_loss


    def infer(self, batch):
        batch = batch.to(self.device)
        y = batch.y
        out, feat = self.model(batch)

        return out, y, batch.n_id, feat