import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, BinaryAUROC, MulticlassAUROC, MulticlassRecall
from torcheval.metrics import MulticlassAUPRC, BinaryAUPRC

from . import ecgfounder

LR = 1e-4
WEIGHT_DECAY = 1e-5


device = torch.device('cuda')

def task2metrics(vocab_size):
    if vocab_size == 1:
        return {
            'accuracy' : BinaryAccuracy(),
            'bacc' : BinaryAccuracy(),
            'auroc' : BinaryAUROC(),
            'auprc' : BinaryAUPRC()
        }
    else:
        return {
            'accuracy' : MulticlassAccuracy(num_classes=vocab_size, average="micro"),
            'bacc' : MulticlassRecall(num_classes=vocab_size),
            'auroc' : MulticlassAUROC(num_classes=vocab_size),
            'auprc' : MulticlassAUPRC(num_classes=vocab_size),
        }

def z_score_normalization(signal):
    mean = signal.mean(axis=(1,2), keepdims=True)
    std = signal.std(axis=(1,2), keepdims=True)
    return (signal - mean) / (std + 1e-8)

class ECGFounderModule():
    def __init__(self, n_outputs, ckpt_path, train_head_only):
        self.n_outputs = n_outputs
        if ckpt_path is None:
            ckpt_path = 'models/ECGFounder/ckpt/12_lead_ECGFounder.pth'

        self.model = ecgfounder.ft_12lead_ECGFounder(
            device, 
            ckpt_path, 
            n_outputs, 
            linear_prob=train_head_only
            )
        
        self.results = {}
        for metric in task2metrics(2).keys():
            self.results[f'train_{metric}'] = []
            self.results[f'val_{metric}'] = []
        
    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        trainloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            num_workers=8, 
            shuffle=True
            )
        valloader = DataLoader(
            validation_dataset, 
            batch_size=batch_size, 
            num_workers=8, 
            shuffle=False
            )

        if self.n_outputs > 2:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='max')
        
        metrics = task2metrics(self.n_outputs)
        for name, metric in metrics.items():
            metric.to(device)

        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs-1}")

            # TRAINING
            self.model.train()
            for name, metric in metrics.items():
                metric.reset()
            for ecg, labels in tqdm(trainloader, desc='Training'):
                ecg = z_score_normalization(ecg)
                ecg = ecg.to(device).float()
                labels = labels.to(device)

                logits = self.model(ecg)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                
                
                if self.n_outputs < 2:
                    logits = torch.sigmoid(logits)
                for name, metric in metrics.items():
                    metric.update(logits, labels.long())

            for name, metric in metrics.items():
                score = metric.compute()
                self.results[f'train_{name}'].append(score.cpu())
                print(f"Training {name} = {score}")

            # VALIDATION
            self.model.eval()
            for name, metric in metrics.items():
                metric.reset()
            with torch.no_grad():
                for ecg, labels in tqdm(valloader, desc='Validation'):
                    ecg = z_score_normalization(ecg)
                    ecg = ecg.to(device).float()
                    labels = labels.to(device)

                    logits = self.model(ecg)
                    if self.n_outputs < 2:
                        logits = torch.sigmoid(logits)
                    for name, metric in metrics.items():
                        metric.update(logits, labels.long())

                for name, metric in metrics.items():
                    score = metric.compute()
                    self.results[f'val_{name}'].append(score.cpu())
                    print(f"Validation {name} = {score}")

            scheduler.step(metrics['auroc'].compute())
