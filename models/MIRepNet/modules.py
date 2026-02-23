import pdb
import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from . import mlm, utils

LR = 0.001
WEIGHT_DECAY = 1e-6
OPTIMIZER = 'adam'
MOMENTUM = 0.9
SCHEDULER = 'cosine'
STEP_SIZE = 30
GAMMA = 0.1
EMB_SIZE = 256
DEPTH = 6

def get_dataloader(dataset, sbj_ids, ch_names, batch_size, shuffle):
    """
    Performs pre-processing on subject level and returns DataLoader for the given dataset
    """
    subjects = np.unique(sbj_ids)
    all_data = []
    all_labels = []
    for s in subjects:
        ix = np.nonzero(sbj_ids[dataset.indices]==s)[0]
        if len(ix) > 0:
            sbj_data, sbj_labels = dataset[ix]
            sbj_data = utils.EA(sbj_data).astype(np.float32)
            sbj_data = utils.pad_missing_channels_diff(sbj_data, utils.target_channels, ch_names)
            all_data.append(sbj_data)
            all_labels.append(sbj_labels)
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    dataset = TensorDataset(
        torch.from_numpy(all_data).float(),  
        torch.from_numpy(all_labels)
    )
    
    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle
    }
    
    return DataLoader(dataset, **loader_args)

class MIRepNetModule():
    def __init__(self, ch_names, n_output, ckpt_path, train_head_only):
        self.train_head_only = train_head_only
        self.is_binary = n_output <= 2
        self.ch_names = ch_names
        if ckpt_path is None:
            ckpt_path = "models/MIRepNet/ckpt/MIRepNet.pth"

        self.model = mlm.mlm_mask(
            emb_size=EMB_SIZE,
            depth=DEPTH,
            n_classes=1 if self.is_binary else n_output,
            pretrainmode=False,
            pretrain=ckpt_path,
            freeze_encoder=self.train_head_only
        ).cuda()

        if self.is_binary:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        self.running_scores = {"train":[], "valid":[]}
        self.results = {'train_accuracy':[], 'train_bacc':[], 'val_accuracy':[], 'val_bacc':[]}

    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train(self, train_loader, optimizer, scheduler):
        self.model.train()
        
        y_true = []
        y_pred = []
        for data, labels in tqdm(train_loader):
            data, labels = data.cuda(), labels.cuda()
            labels = labels.float() if self.is_binary else labels.long()

            _, outputs = self.model(data)
            outputs = outputs.squeeze(-1)
            loss = self.loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = outputs.argmax(dim=-1) if outputs.ndim == 2 else torch.round(torch.sigmoid(outputs))
            y_true += [labels.detach().cpu().numpy()]
            y_pred += [predicted.detach().cpu().numpy()]
        
        scheduler.step()
        
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        self.results['train_accuracy'] += [accuracy_score(y_true, y_pred)]
        self.results['train_bacc'] += [balanced_accuracy_score(y_true, y_pred)]

    def validate(self, val_loader):
        self.model.eval()

        y_true = []
        y_pred = []
        with torch.no_grad():
            for data, labels in tqdm(val_loader):
                data, labels = data.cuda(), labels.cuda()
                _, outputs = self.model(data)
                
                predicted = outputs.argmax(dim=-1) if outputs.ndim == 2 else torch.round(torch.sigmoid(outputs))
                y_true += [labels.detach().cpu().numpy()]
                y_pred += [predicted.detach().cpu().numpy()]

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        self.results['val_accuracy'] += [accuracy_score(y_true, y_pred)]
        self.results['val_bacc'] += [balanced_accuracy_score(y_true, y_pred)]        

    def fit(self, train_dataset, validation_dataset, sbj_ids, batch_size, epochs):
        train_loader = get_dataloader(train_dataset, sbj_ids, self.ch_names, batch_size, shuffle=True)
        val_loader = get_dataloader(validation_dataset, sbj_ids, self.ch_names, batch_size, shuffle=False)

        if self.train_head_only:
            trainable_parameters = [p for n,p in self.model.named_parameters() if 'clshead' in n]
        else:
            trainable_parameters = self.model.parameters()

        if OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                trainable_parameters, 
                lr=LR, 
                weight_decay=WEIGHT_DECAY
            )
        elif OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                trainable_parameters, 
                lr=LR, 
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY
            )
    
        if SCHEDULER == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=epochs
            )
        elif SCHEDULER == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=STEP_SIZE,
                gamma=GAMMA
            )
        else:
            scheduler = None

        for epoch in range(epochs):
            self.train(train_loader, optimizer, scheduler)
            self.validate(val_loader)