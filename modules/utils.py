import numpy as np

def stat_array(x):
    return f'shape: {np.shape(x)} - min: {np.min(x):.2f} - mean: {np.mean(x):.2f} - max: {np.max(x):.2f} - std: {np.std(x):.2f}'


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    