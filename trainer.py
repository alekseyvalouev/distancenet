import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import numpy as np

class ClassificationDistanceNetTrainer:
    def __init__(self, config, dataset, model, train_loader, val_loader):
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def _train_one_epoch(self):
        total_loss, correct, total_n = 0.0, 0, 0

        self.model.train()

        idx = 0
        for img_1, img_2, y in tqdm(self.train_loader, desc="Training"):
            img_1, img_2, y = img_1.to(self.device), img_2.to(self.device), y.type(torch.long).to(self.device)
            self.optimizer.zero_grad()
            out = self.model(img_1, img_2)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size

            output_class = torch.argmax(out, dim=1)
            correct += (output_class == y).sum().item()
            total_n += batch_size

            idx += batch_size

        avg_loss = total_loss / total_n
        acc = correct / total_n
        
        return avg_loss, acc
    
    def _evaluate(self):
        total_loss, correct, total_n = 0.0, 0, 0

        self.model.eval()

        with torch.no_grad():
            idx = 0
            for img_1, img_2, y in tqdm(self.val_loader, desc="Validation"):
                img_1, img_2, y = img_1.to(self.device), img_2.to(self.device), y.type(torch.long).to(self.device)
                out = self.model(img_1, img_2)
                loss = self.criterion(out, y)

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size

                output_class = torch.argmax(out, dim=1)
                correct += (output_class == y).sum().item()
                total_n += batch_size

                idx += batch_size

            avg_loss = total_loss / total_n
            acc = correct / total_n

        return avg_loss, acc

class RegressionDistanceNetTrainer:
    def __init__(self, config, dataset, model, train_loader, val_loader, device=None):
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        self.criterion_none = torch.nn.MSELoss(reduction='none')
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def _train_one_epoch(self):
        total_loss, total_n = 0.0, 0
        class_loss_sum = defaultdict(float)
        class_count = defaultdict(int)

        self.model.train()

        for img_1, img_2, y in tqdm(self.train_loader, desc="Training"):
            img_1, img_2, y = img_1.to(self.device), img_2.to(self.device), y.type(torch.float).to(self.device)
            self.optimizer.zero_grad()
            out = self.model(img_1, img_2).squeeze(-1)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size

            # Per-class loss tracking
            with torch.no_grad():
                per_sample_loss = self.criterion_none(out, y)
                for cls in y.unique():
                    mask = y == cls
                    cls_key = cls.item()
                    class_loss_sum[cls_key] += per_sample_loss[mask].sum().item()
                    class_count[cls_key] += mask.sum().item()

            total_n += batch_size

        avg_loss = total_loss / total_n
        per_class_losses = {cls: class_loss_sum[cls] / class_count[cls] for cls in sorted(class_loss_sum)}
        
        return avg_loss, per_class_losses
    
    def _evaluate(self):
        total_loss, total_n = 0.0, 0
        class_loss_sum = defaultdict(float)
        class_count = defaultdict(int)

        self.model.eval()

        with torch.no_grad():
            for img_1, img_2, y in tqdm(self.val_loader, desc="Validation"):
                img_1, img_2, y = img_1.to(self.device), img_2.to(self.device), y.type(torch.float).to(self.device)
                out = self.model(img_1, img_2).squeeze(-1)
                loss = self.criterion(out, y)

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size

                # Per-class loss tracking
                per_sample_loss = self.criterion_none(out, y)
                for cls in y.unique():
                    mask = y == cls
                    cls_key = cls.item()
                    class_loss_sum[cls_key] += per_sample_loss[mask].sum().item()
                    class_count[cls_key] += mask.sum().item()

                total_n += batch_size

            avg_loss = total_loss / total_n
            per_class_losses = {cls: class_loss_sum[cls] / class_count[cls] for cls in sorted(class_loss_sum)}

        return avg_loss, per_class_losses

class ViNT_DistanceNetTrainer:
    def __init__(self, config, dataset, model, train_loader, val_loader, device=None):
        self.dataset = dataset
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss()
        self.criterion_none = torch.nn.MSELoss(reduction='none')
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def _train_one_epoch(self):
        total_loss, total_n = 0.0, 0
        class_loss_sum = defaultdict(float)
        class_count = defaultdict(int)

        self.model.train()

        for img_1, img_2, y in tqdm(self.train_loader, desc="Training"):
            img_1, img_2, y = img_1.to(self.device), img_2.to(self.device), y.type(torch.float).to(self.device)
            self.optimizer.zero_grad()
            out, _ = self.model(img_1, img_2)
            out = out.squeeze(-1)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size

            # Per-class loss tracking
            with torch.no_grad():
                per_sample_loss = self.criterion_none(out, y)
                for cls in y.unique():
                    mask = y == cls
                    cls_key = cls.item()
                    class_loss_sum[cls_key] += per_sample_loss[mask].sum().item()
                    class_count[cls_key] += mask.sum().item()

            total_n += batch_size

        avg_loss = total_loss / total_n
        per_class_losses = {cls: class_loss_sum[cls] / class_count[cls] for cls in sorted(class_loss_sum)}
        
        return avg_loss, per_class_losses
    
    def _evaluate(self):
        total_loss, total_n = 0.0, 0
        class_loss_sum = defaultdict(float)
        class_count = defaultdict(int)

        self.model.eval()

        with torch.no_grad():
            for img_1, img_2, y in tqdm(self.val_loader, desc="Validation"):
                img_1, img_2, y = img_1.to(self.device), img_2.to(self.device), y.type(torch.float).to(self.device)
                out, _ = self.model(img_1, img_2)
                out = out.squeeze(-1)
                loss = self.criterion(out, y)

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size

                # Per-class loss tracking
                per_sample_loss = self.criterion_none(out, y)
                for cls in y.unique():
                    mask = y == cls
                    cls_key = cls.item()
                    class_loss_sum[cls_key] += per_sample_loss[mask].sum().item()
                    class_count[cls_key] += mask.sum().item()

                total_n += batch_size

            avg_loss = total_loss / total_n
            per_class_losses = {cls: class_loss_sum[cls] / class_count[cls] for cls in sorted(class_loss_sum)}

        return avg_loss, per_class_losses