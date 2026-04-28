import torch
import torch.nn.functional as F
from utils.utils import format_time, compute_metrics
from tqdm import tqdm
import time

class Trainer:
    def __init__(self, model, trainloader,
        val_loader, test_loader, epochs, optimizer, 
        early_stopping=None, device=None, scheduler=None,
        use_amp=True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.optimizer = optimizer
        self.early_stopping = early_stopping
        self.scheduler = scheduler

        # Hyperparameters for semi-supervised learning
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)


    def training(self):
        total_start = time.perf_counter()
        assert self.early_stopping is not None, "early_stopping must be provided."
        assert self.val_loader is not None, "val_loader must be provided."

        for epoch in range(self.epochs):
            train_loss, train_metrics = self.train_one_epoch(epoch)
            val_loss, val_metrics = self.evaluate(data_loader=self.val_loader, loader_name="Validation Evaluating...")
            self._print_epoch_summary(epoch, train_loss, train_metrics, val_loss, val_metrics)
            
            # Early stopping check
            self.early_stopping(val_metrics["f1"], self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered...")
                break

        # Computational time
        total_time = time.perf_counter() - total_start
        print(f"Total training time: {format_time(total_time)}")

        # Load best model
        self.early_stopping.load_best_model(self.model, self.device)
        print("Loaded best model.")
        

        # Best model evaluation(train, val, test)
        self._evaluate_and_log(self.trainloader, "Best Model Train")
        self._evaluate_and_log(self.val_loader, "Best Model Validation")
        self._evaluate_and_log(self.test_loader, "Test")


    def train_one_epoch(self, epoch=None):
        self.model.train()

        total_loss = 0.0
        all_labels, all_preds = [], []

        desc = "Training" if epoch is None else f"Training {epoch + 1}/{self.epochs}"
        progress_bar = tqdm(self.trainloader, desc=desc, leave=False)

        for step, (x, labels) in enumerate(progress_bar, start=1):
            x = x.to(self.device)
            labels = labels.to(self.device).long()

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix(loss=f"{total_loss / step:.4f}")
        
        if self.scheduler is not None:
            self.scheduler.step()
            
        avg_loss = total_loss / len(self.trainloader)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics



    def evaluate(self, data_loader, loader_name="Evaluating..."):
        self.model.eval()
        total_loss = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for x, labels in tqdm(data_loader, desc=loader_name, leave=False):
                x = x.to(self.device)
                labels = labels.to(self.device).long()

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(x)
                    loss = F.cross_entropy(outputs, labels)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                total_loss += loss.item()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics



    def _evaluate_and_log(self, data_loader, title):
        loss, metrics = self.evaluate(data_loader=data_loader, loader_name=f"{title} Evaluating...")
        print(
            f"[{title}] Loss: {loss:.4f}, "
            f"Acc: {metrics['accuracy']:.4f}, "
            f"Prec: {metrics['precision']:.4f}, "
            f"Rec: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )
        print(f"[{title}] Classification Report:\n{metrics['classification_report']}")
        print(f"[{title}] Confusion Matrix:\n{metrics['confusion_matrix']}\n")

    def _print_epoch_summary(self, epoch, train_loss, train_metrics, val_loss, val_metrics):
        print(
            f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}, "
            f"Acc: {train_metrics['accuracy']:.4f}, "
            f"Prec: {train_metrics['precision']:.4f}, "
            f"Rec: {train_metrics['recall']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}"
        )

        print(
            f"[Epoch {epoch + 1}] Val Loss: {val_loss:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}, "
            f"Prec: {val_metrics['precision']:.4f}, "
            f"Rec: {val_metrics['recall']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}"
        )