import copy
import math
import os
import time
import torch.nn as nn
import numpy as np
import torch
from loguru import logger as log

from utils.metric import All_Metrics


class Trainer(object):
    def __init__(self, model, scaler, args):
        super(Trainer, self).__init__()
        self.model = model
        self.scaler = scaler
        self.args = args
        self._init_training()
        # self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')

    def _init_training(self):
        if self.args.loss_func == "mae":
            self.loss = torch.nn.L1Loss(reduction="sum").to(self.args.device)
        elif self.args.loss_func == "mse":
            self.loss = torch.nn.MSELoss().to(self.args.device)
        else:
            raise ValueError
        
        # configure megacrn
        if self.args.model_name == "megacrn":
            self.separate_loss = nn.TripletMarginLoss(margin=1.0)
            self.compact_loss = nn.MSELoss()

        if self.args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                eps=1.0e-8,
                weight_decay=self.args.weight_decay,
                amsgrad=False
            )

        if self.args.lr_decay:
            print("Applying learning rate decay.")
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.args.lr_decay_step,
                gamma=self.args.lr_decay_rate,
            )

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., : self.args.input_dim].to(self.args.device)
                label = target[..., : self.args.output_dim].to(self.args.device)
                if self.args.model_name == "megacrn":
                    output, query, pos, neg = self.model(data, None)
                else:
                    output = self.model(data, None)
                if not self.args.is_norm_loss:
                    label = self.scaler.inverse_transform(label)
                    output = self.scaler.inverse_transform(output)
                loss = self.loss(output.cuda(), label) / (
                    data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3]
                )

                # a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                else:
                    log.info("The loss of one batch contains nan value.")

        val_loss = total_val_loss / len(val_dataloader)
        log.info(
            "**********Val Epoch {}: average Loss: {:.6f}".format(
                epoch, val_loss
            )
        )
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            if self.args.is_oracle or self.args.is_last or (self.args.incre_num == 1):
                data, target = data
            else:
                data, target, mask = data
                mask = mask.to(self.args.device)
            
            data = data[..., : self.args.input_dim].to(self.args.device)
            label = target[..., : self.args.output_dim].to(self.args.device)
            self.optimizer.zero_grad()

            # data and target shape: B, T, N, F; output shape: B, T, N, F
            if self.args.is_oracle or self.args.is_last or (self.args.incre_num == 1):
                if self.args.model_name == "megacrn":
                    output, query, pos, neg = self.model(data, None)
                else:
                    output = self.model(data, None)
            else:
                if self.args.model_name == "megacrn":
                    output, query, pos, neg = self.model(data, None)
                else:
                    output = self.model(data, None)
  
            if not self.args.is_norm_loss:
                label = self.scaler.inverse_transform(label)
                output = self.scaler.inverse_transform(output)

            # do mask loss
            if self.args.is_mask_loss:
                mask = mask.unsqueeze(-1).repeat(1, 1, output.size(1)).unsqueeze(-1).transpose(1, 2) # [batch, T, N, 1]
                output *= mask
                label *= mask
                loss = self.loss(output.cuda(), label)
                loss = loss / torch.sum(mask)
            else:
                loss = self.loss(output.cuda(), label)
                loss = loss / data.numel()
            if self.args.model_name == "megacrn":
                loss2 = self.separate_loss(query, pos, neg)
                loss3 = self.compact_loss(query, pos)
                loss = loss + self.args.lamb * loss2 + self.args.lamb1 * loss3
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
            self.optimizer.step()
            total_loss += loss.item()

            # log information
            if batch_idx % self.args.log_step == 0:
                log.info(
                    "Train Epoch {}: {}/{} Loss: {:.6f}".format(
                        epoch, batch_idx, self.train_per_epoch, loss.item()
                    )
                )
        train_epoch_loss = total_loss / self.train_per_epoch
        log.info(
            "**********Train Epoch {}: averaged Loss: {:.6f}".format(
                epoch, train_epoch_loss
            )
        )

        # learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self, train_loader, val_loader):
        self.train_per_epoch = len(train_loader)
        self.train_loader = train_loader
        best_loss = float("inf")
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            # epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            # print(time.time()-epoch_time)

            val_epoch_loss = self.val_epoch(epoch, val_loader)

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                log.warning("Gradient explosion detected. Ending...")
                break

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stoping:
                if not_improved_count == self.args.patience:
                    log.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.args.patience)
                    )
                    self.model.load_state_dict(best_model)
                    break
            # save the best state
            if best_state == True:
                log.info(
                    "*********Current best model saved!*********************"
                )
                best_model = copy.deepcopy(self.model.state_dict())

        training_time = time.time() - start_time
        log.info(
            "Total training time: {:.4f}min, best loss: {:.6f}".format(
                (training_time / 60), best_loss
            )
        )

        return self.model

    def save_checkpoint(self):
        state = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.args,
        }
        torch.save(state, self.best_path)
        log.info("Saving current best model to " + self.best_path)

    def test(self, test_loader):
        self.model.eval()
        y_pred = []
        y_true = []
        # features = [] # visualization
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data[..., : self.args.input_dim].to(self.args.device)
                label = target[..., : self.args.output_dim].to(self.args.device)
                if self.args.model_name == "megacrn":
                    output, query, pos, neg = self.model(data, None)
                else:
                    output = self.model(data, None)
                    # output, feature = self.model(data, None) # visualization
                    # features.append(feature)
                y_true.append(label)
                y_pred.append(output)
        # features = np.concatenate(features, axis=0)
        # np.save("all_feature.npy", features)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        
        if not self.args.is_norm_metric:
            y_true = self.scaler.inverse_transform(y_true)
            y_pred = self.scaler.inverse_transform(y_pred)

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(
                y_pred[:, t, ...],
                y_true[:, t, ...],
                None,
                None,
            )
            log.info(
                "Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                    t + 1, mae, rmse, mape * 100
                )
            )
        mae, rmse, mape = All_Metrics(
            y_pred, y_true, None, None
        )

        log.info(
            "Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                mae, rmse, mape * 100
            )
        )
        mae, rmse, mape = All_Metrics(
            y_pred[:, :, :self.args.first_num_nodes,:], y_true[:, :, :self.args.first_num_nodes,:], None, None
        )
        log.info(
            "Average of continual variables, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                mae, rmse, mape * 100
            )
        )
        mae, rmse, mape = All_Metrics(
            y_pred[:, :, self.args.first_num_nodes:,:], y_true[:, :, self.args.first_num_nodes:,:], None, None
        )
        log.info(
            "Average of expanding variables, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                mae, rmse, mape * 100
            )
        )
        
        return "", y_true, y_pred

    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))
