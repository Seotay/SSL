import random
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes} min {seconds:.2f} sec"    

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path="./checkpoint/best_model.pt"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_val_f1 = None
        self.early_stop = False

    def __call__(self, current_val_f1, model):
        if self.best_val_f1 is None:
            self.best_val_f1 = current_val_f1
            self.save_checkpoint(model)

        elif current_val_f1 >= self.best_val_f1 + self.delta:
            self.save_checkpoint(model)
            self.best_val_f1 = current_val_f1
            self.counter = 0

        else:
            self.counter += 1
            if self.verbose:
                print(f"\tEarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        if self.verbose and self.best_val_f1 is not None:
            print(f"\tValidation F1 improved → Saving model: {self.path}")
        torch.save(model.state_dict(), self.path)

    def load_best_model(self, model, device=None):
        map_location = device if device is not None else "cpu"
        state_dict = torch.load(self.path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "classification_report": report, "confusion_matrix": conf_matrix}