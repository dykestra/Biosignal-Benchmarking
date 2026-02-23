import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from fairseq_signals.models import build_model_from_checkpoint
from fairseq_signals.models.classification.ecg_transformer_classifier import ECGTransformerClassificationModel
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, BinaryAUROC, MulticlassAUROC, MulticlassRecall
from torcheval.metrics import MulticlassAUPRC, BinaryAUPRC

LR = 1e-6
BETAS = (0.9, 0.98)

device = torch.device('cuda')

def z_score_normalization(signal):
    mean = signal.mean(axis=(1,2), keepdims=True)
    std = signal.std(axis=(1,2), keepdims=True)
    return (signal - mean) / (std + 1e-8)

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
            'bacc' : MulticlassAccuracy(num_classes=vocab_size, average="macro"),
            'auroc' : MulticlassAUROC(num_classes=vocab_size),
            'auprc' : MulticlassAUPRC(num_classes=vocab_size),
        }

class ECG_FMModule():
    def __init__(self, n_outputs, ckpt_path, train_head_only=False):
        self.n_outputs = n_outputs
        self.train_head_only = train_head_only

        if ckpt_path is None:
            ckpt_path = "/opt/ml/users/na/repos/ecg-fm/ckpts/mimic_iv_ecg_physionet_pretrained.pt"
        ft_ckpt_path = "/opt/ml/users/na/repos/ecg-fm/ckpts/mimic_iv_ecg_finetuned.pt"

        pt_ckpt = torch.load(ckpt_path)

        ft_model: ECGTransformerClassificationModel = build_model_from_checkpoint(checkpoint_path=ft_ckpt_path)
        tgt_sd = ft_model.encoder.state_dict()
        filtered = {k: v for k, v in pt_ckpt['model'].items()
                    if k in tgt_sd and v.shape == tgt_sd[k].shape}

        missing, unexpected = ft_model.encoder.load_state_dict(filtered, strict=False)

        print("Loaded keys:", len(filtered))
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        ft_model.proj = nn.Linear(ft_model.proj.in_features, n_outputs)
        self.model = ft_model.to(device)

        if self.train_head_only:
            for n, p in self.model.named_parameters():
                p.requires_grad = (n == 'proj')
        
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

        metrics = task2metrics(self.n_outputs)
        for name, metric in metrics.items():
            metric.to(device)    

        if self.train_head_only:
            optimizer = optim.Adam(self.model.proj.parameters(), lr=LR, betas=BETAS)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=LR, betas=BETAS)

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

                logits = self.model(source=ecg)['out']
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

                    logits = self.model(source=ecg)['out']
                if self.n_outputs < 2:
                        logits = torch.sigmoid(logits)
                for name, metric in metrics.items():
                    metric.update(logits, labels.long())

                for name, metric in metrics.items():
                    score = metric.compute()
                    self.results[f'val_{name}'].append(score.cpu())
                    print(f"Validation {name} = {score}")
