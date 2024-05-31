class EarlyStopping:
    def __init__(self, patience: int = 3, delta: float = 0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        model.save(".cache/checkpoint.pt")

    def load_checkpoint(self, model):
        model.load(".cache/checkpoint.pt")


class L1Regularization:
    def __init__(self, weight_decay=1e-5):
        self.weight_decay = weight_decay

    def __call__(self, parameters) -> float:
        norm = sum(p.abs().sum() for p in parameters)
        return norm * self.weight_decay
