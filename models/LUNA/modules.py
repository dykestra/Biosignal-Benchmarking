from tqdm import tqdm
import math
import numpy as np
import mne
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, BinaryRecall, BinarySpecificity, BinaryAUROC, MulticlassAUROC, MulticlassRecall, MulticlassConfusionMatrix, BinaryConfusionMatrix
from torchmetrics import Metric
from torcheval.metrics import MulticlassAUPRC, BinaryAUPRC
import pdb

from .LUNA import LUNA

patch_size=40

class WarmupCosineScheduler:
    def __init__(self, optimizer, max_epochs, warmup_epochs, lr, min_lr=2.5e-6, warmup_lr_init=2.5e-7):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_lr_init = warmup_lr_init

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            alpha = epoch / max(1, self.warmup_epochs)
            lr = self.warmup_lr_init + alpha * (self.lr - self.warmup_lr_init)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.max_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        return lr

class BinaryBalancedAccuracy(Metric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.recall = BinaryRecall(threshold=threshold)
        self.spec = BinarySpecificity(threshold=threshold)

    def update(self, preds, target):
        self.recall.update(preds, target)
        self.spec.update(preds, target)

    def compute(self):
        return (self.recall.compute() + self.spec.compute()) / 2
    
    def reset(self):
        self.recall.reset()
        self.spec.reset()

def z_score_normalization(x, eps=1e-8):
    return (x - x.mean(dim=-1, keepdim=True)) / (
        x.std(dim=-1, keepdim=True).clamp_min(eps)
    )

class LUNAModule():
    def __init__(self, ch_names, n_out, ckpt_path, train_head_only):
        self.chanlocs = self.get_chanlocs(ch_names)
        self.n_out = n_out
        self.train_head_only = train_head_only
        
        self.model = LUNA(patch_size=patch_size,
                          embed_dim=64,
                          num_heads=2,
                          depth=8,
                          num_queries=4,
                          drop_path=0.1,
                          num_classes=n_out)
        if ckpt_path is None:
            ckpt_path = "models/LUNA/ckpt/LUNA_base.safetensors"
        state = load_file(ckpt_path)

        cleaned = OrderedDict()
        for k, v in state.items():
            k = k.removeprefix("model.")
            if any(skip in k for skip in ["decoder_head", "channel_emb", "classifier"]):
                continue
            cleaned[k] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        if self.train_head_only:
            for name, param in self.model.named_parameters():
                if name != 'classifier':
                    param.requires_grad = False

        if self.n_out > 2:
            self.metrics = {
                    'accuracy' : MulticlassAccuracy(num_classes=self.n_out, average="micro"),
                    'bacc' : MulticlassRecall(num_classes=self.n_out, average='macro')
                }
        else:
            self.metrics = {
                'accuracy' : BinaryAccuracy(),
                'bacc' : BinaryBalancedAccuracy()
            }

        self.results = {}
        for m in self.metrics.keys():
            self.results[f'train_{m}'] = []
            self.results[f'val_{m}'] = []


    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    def get_chanlocs(self, channel_names):
        montage = mne.channels.make_standard_montage('standard_1005')

        pos_dict = {name.upper(): pos for name, pos in montage.get_positions()['ch_pos'].items()}

        chanlocs = []
        missing = []

        for c in channel_names:
            if c not in pos_dict:
                missing.append(c)
            else:
                chanlocs.append(pos_dict[c])

        if missing:
            raise ValueError(f'Could not find channels {missing} in standard_1005 montage')

        return torch.tensor(np.asarray(chanlocs), dtype=torch.float32)

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        device = torch.device('cuda')
        self.model = self.model.to(device)

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

        chanlocs = self.chanlocs.unsqueeze(0).expand(batch_size, -1, -1) # repeat chanlocs for the batch
        chanlocs = chanlocs.to(device)

        Y = [y for _, y in train_dataset]
        class_counts = torch.bincount(torch.tensor(Y).long(), minlength=self.n_out)
        class_weights = (class_counts.sum() / (self.n_out * class_counts)).float().to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5.0e-4,
            betas=(0.9, 0.999),
            weight_decay=0.05
            )
        scheduler = WarmupCosineScheduler(
            optimizer,
            max_epochs=epochs,
            warmup_epochs=10,
            lr=5.0e-4,
            min_lr=2.5e-6,
            warmup_lr_init=2.5e-7,
        )

        for name, metric in self.metrics.items():
            metric.to(device)

        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs-1}")

            # TRAINING
            lr = scheduler.step(epoch)
            print(f"LR={lr}")
            self.model.train()
            for name, metric in self.metrics.items():
                metric.reset()
            for eeg, labels in tqdm(trainloader, desc='Training'):
                B, C, T = eeg.shape
                S = T // patch_size
                eeg = eeg[...,:S*patch_size] # crop to patch size
                eeg = z_score_normalization(eeg)
                eeg = eeg.to(device).float()
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits, _ = self.model(eeg, channel_locations=chanlocs[:B])
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()                

                preds = logits.argmax(dim=1)
                for name, metric in self.metrics.items():
                    metric.update(preds, labels.long())

            for name, metric in self.metrics.items():
                score = metric.compute()
                macro = score.mean()
                value = macro.detach().cpu().item()
                self.results[f'train_{name}'].append(value)
                print(f"Training {name} = {score}")

            # VALIDATION
            self.model.eval()
            if self.n_out > 2:
                confusion_matrix = MulticlassConfusionMatrix(num_classes=self.n_out).cuda()
            else:
                confusion_matrix = BinaryConfusionMatrix().cuda()
            for name, metric in self.metrics.items():
                metric.reset()
            with torch.no_grad():
                for eeg, labels in tqdm(valloader, desc='Validation'):
                    B, C, T = eeg.shape
                    S = T // patch_size
                    eeg = eeg[...,:S*patch_size] # crop to patch size
                    eeg = z_score_normalization(eeg)
                    eeg = eeg.to(device).float()
                    labels = labels.to(device).long()

                    logits, _ = self.model(eeg, channel_locations=chanlocs[:B])

                    preds = logits.argmax(dim=1)
                    confusion_matrix.update(preds, labels)
                    for name, metric in self.metrics.items():
                        metric.update(preds, labels)

                for name, metric in self.metrics.items():
                    score = metric.compute()
                    macro = score.mean()
                    value = macro.detach().cpu().item()
                    self.results[f'val_{name}'].append(value)
                    print(f"Validation {name} = {score}")

                print(confusion_matrix.compute())
