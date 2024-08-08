from baseTrainer import baseTrainer


class Trainer(baseTrainer):
    def __init__(self, model, scaler, cfg):
        super().__init__(model, scaler, cfg)

    def forward(self, batch):
        X, y, idx = batch
        X = X.to(self.device)
        y = y.to(self.device)

        out = self.model(X)

        return out, y, idx
