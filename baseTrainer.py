import copy

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from loguru import logger as log
from torchmetrics.regression.mae import MeanAbsoluteError as MAE
from torchmetrics.regression.mape import MeanAbsolutePercentageError as MAPE
from torchmetrics.regression.mse import MeanSquaredError as MSE
from tqdm import tqdm

from utils.debug_tool import p_profiler, performance_profiler_time
from utils.metric import All_Metrics


def eval_metric(pred, true, prefix=""):
    mae, rmse, mape = All_Metrics(pred, true, None, None)
    metrics = {
        f"{prefix}_rmse": rmse,
        f"{prefix}_mae": mae,
        f"{prefix}_mape": mape,
    }

    return metrics


def _log_metric(metrics, prefix=""):
    _log = "\n"
    # log header
    _log += f"{prefix} \n"
    template = "{0:>20}|{1:<20}"
    _log += "-" * 40 + "\n"
    _log += template.format("Metrics", "Metrics") + "\n"
    _log += "-" * 40 + "\n"
    for name, metric in metrics.items():
        _log += template.format(name, metric) + "\n"
    _log += f"{'-'*40} \n"
    return _log


class baseTrainer:
    def __init__(self, model, scaler, cfg):
        self.model = model
        self.device = cfg.device
        self.skip_epoch = cfg.skip_epoch
        self.epochs = cfg.epochs
        self.early_stoping = cfg.early_stoping
        self.patience = cfg.patience
        self.save_path = cfg.log_path
        self.is_norm_metric = cfg.is_norm_metric
        self.is_norm_loss = cfg.is_norm_loss
        self.is_cons_loss = cfg.is_cons_loss
        self.cl_loss_weight = cfg.cl_loss_weight
        self.scaler = scaler
        self.first_feats = cfg.first_feats
        self.is_col = cfg.is_col
        self.loss_func = torch.nn.L1Loss()
        self.optimizer = cfg.optimizer
        if cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )

    def forward(self, batch):
        raise NotImplementedError

    # @p_profiler
    def train(self, train_dataset, valid_dataset):
        self.model.train()
        optimizer = self.optimizer

        _loss = 0
        _v_metric = {name: 0 for name in ["rmse", "mae", "mape"]}
        best_loss = float("inf")
        not_improved_count = 0
        best_model = None
        for epoch in range(self.epochs):
            train_loss = []
            with tqdm(train_dataset, desc=f"Epoch {epoch}") as tq:
                _log = {}
                for batch in tq:
                    optimizer.zero_grad()

                    if self.is_cons_loss:
                        out, y, idx, cons_loss = self.forward(batch)
                    else:
                        out, y, idx = self.forward(batch)

                    if not self.is_norm_loss:
                        if self.is_col:
                            out = self.scaler.inverse_transform_col(
                                out, batch.ptr
                            )
                            y = self.scaler.inverse_transform_col(y, batch.ptr)
                        else:
                            out = self.scaler.inverse_transform(out)
                            y = self.scaler.inverse_transform(y)
                    
                    loss = self.loss_func(out, y)
                    # if epoch > 5:
                    if self.is_cons_loss:
                        loss = self.loss_func(out, y) + self.cl_loss_weight * cons_loss

                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.detach().item())

                    _loss = {"loss": np.mean(train_loss)}
                    _log = {**_loss, **_v_metric}

                    tq.set_postfix(_log)

                if epoch % self.skip_epoch == 0 and valid_dataset is not None:
                    log.info(f"Epoch {epoch} result is: {_log}")
                    _v_metric = self.valid(valid_dataset)

                if self.early_stoping:
                    if _v_metric["mae"] < best_loss:
                        best_loss = _v_metric["mae"]
                        best_model = copy.deepcopy(self.model.state_dict())
                        not_improved_count = 0
                    else:
                        not_improved_count += 1
                        if not_improved_count >= self.patience:
                            log.info(f"Stop training at epoch {epoch}")
                            self.model.load_state_dict(best_model)
                            break
        return self.model

    def valid(self, valid_dataset):
        metrics = {
            "rmse": MSE(squared=False),
            "mae": MAE(),
            "mape": MAPE(),
        }

        with torch.no_grad():
            for batch in valid_dataset:
                if self.is_cons_loss:
                    out, y, idx, cons_loss = self.infer(batch)
                else:
                    out, y, idx = self.forward(batch)

                out, y, idx = (
                    out.cpu().detach(),
                    y.cpu().detach(),
                    idx.cpu().detach(),
                )

                if not self.is_norm_metric:
                    if self.is_col:
                        out = self.scaler.inverse_transform_col(out, batch.ptr)
                        y = self.scaler.inverse_transform_col(y, batch.ptr)
                    else:
                        out = self.scaler.inverse_transform(out)
                        y = self.scaler.inverse_transform(y)

                for name, metric in metrics.items():
                    metric.update(out, y)
            for name, metric in metrics.items():
                metrics[name] = metric.compute().item()

            _log = " ".join(
                [f"{name}: {metric}" for name, metric in metrics.items()]
            )
            log.info(_log)
        return metrics

    def test(self, test_dataset):
        self.model.eval()

        metrics = {
            "rmse": MSE(squared=False),
            "mae": MAE(),
            "mape": MAPE(),
        }
        real_y, pred_y, feat_idx = [], [], []
        features = []
        with torch.no_grad():
            for batch in test_dataset:
                if self.is_cons_loss:
                    out, y, idx, feat = self.infer(batch)
                    features.append(feat)
                else:
                    out, y, idx = self.forward(batch)
                out, y, idx = (
                    out.cpu().detach(),
                    y.cpu().detach(),
                    idx.cpu().detach(),
                )

                if not self.is_norm_metric:
                    if self.is_col:
                        out = self.scaler.inverse_transform_col(out, batch.ptr)
                        y = self.scaler.inverse_transform_col(y, batch.ptr)
                    else:
                        out = self.scaler.inverse_transform(out)
                        y = self.scaler.inverse_transform(y)
                real_y.append(copy.deepcopy(y.numpy()))
                pred_y.append(copy.deepcopy(out.numpy()))
                feat_idx.append(copy.deepcopy(idx.numpy()))

                for name, metric in metrics.items():
                    metric.update(out, y)
            for name, metric in metrics.items():
                metrics[name] = metric.compute().item()

            real_y = np.concatenate(real_y, axis=0)
            pred_y = np.concatenate(pred_y, axis=0)

            features = np.concatenate(features, axis=0)           
            np.save("cons_all_feature.npy", features)

            feat_idx = np.concatenate(feat_idx, axis=0)
            y, p = real_y, pred_y
            mask = feat_idx < self.first_feats

            re_m = np.where(mask)[0]
            new_m = np.where(~mask)[0]
            metrics = {
                **metrics,
                **eval_metric(p, y, prefix="all"),
                **eval_metric(p[re_m], y[re_m], prefix="remain"),
                **eval_metric(p[new_m], y[new_m], prefix="new"),
            }

            _log = _log_metric(metrics, prefix="Testloader result is: ")
            log.info(_log)
        return metrics, real_y, pred_y
