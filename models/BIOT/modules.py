import pdb
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from . import biot

# BIOT hparams
LR=1e-3
WEIGHT_DECAY=1e-5

TARGET_CHANNELS = ["FP1", "F7", "T7", "P7", "FP2", "F8", "T8", "P8", "FP1", "F3", "C3", "P3", "FP2", "F4", "C4", "P4", "C3", "C4"]

def reorder_channels(data, source_channels):
    # pdb.set_trace()
    reordered = torch.zeros((data.shape[0], len(TARGET_CHANNELS), data.shape[-1]))
    for target_idx, label in enumerate(TARGET_CHANNELS):
        if label in source_channels:
            mapped_idx = np.where(source_channels == label)[0]
            reordered[:, target_idx, :] = data[:, mapped_idx, :].squeeze()
    return reordered.cuda()

# (from utils.py) define binary cross entropy loss
def BCE(y_hat, y):
    # y_hat: (N, 1)
    # y: (N, 1)
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    loss = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )
    return loss.mean()


class BIOTModule(pl.LightningModule):
    def __init__(self, ch_names, n_outputs, ckpt_path, train_head_only=False, in_channels=18, token_size=200, hop_length=100):
        super().__init__()
        self.ch_names = ch_names
        self.n_classes = n_outputs if n_outputs > 2 else 1
        self.model = biot.BIOTClassifier(
            n_channels=in_channels,
            n_fft=token_size,
            hop_length=hop_length,
            n_classes=self.n_classes
        )
        if ckpt_path is None:
            ckpt_path = "models/BIOT/ckpt/EEG-six-datasets-18-channels.ckpt"
        self.model.biot.load_state_dict(torch.load(ckpt_path))

        self.train_head_only = train_head_only
        if self.train_head_only:
            for param in self.model.biot.parameters():
                param.requires_grad = False

        self.threshold = 0.5
        self.results = {'train_accuracy':[], 'train_bacc':[], 'val_accuracy':[], 'val_bacc':[]}

        self.validation_step_outputs = []
        self.validation_step_gts = []
        self.train_step_preds = []
        self.train_step_gts = []

    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def training_step(self, batch, batch_idx):
        X, y = batch
        X = reorder_channels(X, self.ch_names)

        prob = self.model(X.float())
        if self.n_classes > 1:
            loss = nn.CrossEntropyLoss()(prob, y)
        else:
            loss = BCE(prob, y)

        self.train_step_preds.append(torch.sigmoid(prob).detach().cpu())
        self.train_step_gts.append(y.detach().cpu())
        return loss
    
    def on_train_epoch_end(self):
        train_step_preds = torch.cat(self.train_step_preds).squeeze()
        train_step_gts = torch.cat(self.train_step_gts)

        if self.n_classes == 1:
            preds = np.round(train_step_preds)
        else:
            preds = np.argmax(train_step_preds, axis=1)
        gts = train_step_gts.reshape(preds.shape)

        self.results['train_accuracy'].append(accuracy_score(gts, preds))
        self.results['train_bacc'].append(balanced_accuracy_score(gts, preds))
        self.train_step_preds.clear()
        self.train_step_gts.clear()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X = reorder_channels(X, self.ch_names)
        with torch.no_grad():
            prob = self.model(X)
            if self.n_classes < 2:
                step_result = torch.sigmoid(prob).cpu()
            else:
                step_result = prob.cpu()
            step_gt = y.cpu()

        self.validation_step_outputs.append(step_result)
        self.validation_step_gts.append(step_gt)
        return step_result, step_gt

    def on_validation_epoch_end(self):
        val_step_outputs = torch.cat(self.validation_step_outputs)
        val_step_gts = torch.cat(self.validation_step_gts)

        if self.n_classes == 1:
            preds = np.round(val_step_outputs)
        else:
            preds = np.argmax(val_step_outputs, axis=1)
        gts = val_step_gts.reshape(preds.shape)

        self.results['val_accuracy'].append(accuracy_score(gts, preds))
        self.results['val_bacc'].append(balanced_accuracy_score(gts, preds))

        self.validation_step_outputs.clear() 
        self.validation_step_gts.clear()

    def configure_optimizers(self):
        if self.train_head_only:
            optimizer = torch.optim.Adam(
                self.model.classifier.parameters(),
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )
        else:           
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=LR,
                weight_decay=WEIGHT_DECAY,
            )

        return [optimizer]