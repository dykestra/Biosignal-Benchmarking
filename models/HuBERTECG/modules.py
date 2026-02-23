import pdb
import numpy as np
import torch
import pickle
import sys
from math import ceil
from typing import Tuple, Any
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.amp as amp
import torch.optim as optim
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy, BinaryAUROC, MulticlassAUROC, BinaryRecall, BinarySpecificity, MulticlassRecall, MulticlassConfusionMatrix
from torchmetrics import Metric
from torcheval.metrics import MulticlassAUPRC, BinaryAUPRC
from transformers import get_linear_schedule_with_warmup

from .hubert_ecg import HuBERTECG as HuBERT, HuBERTECGConfig
from .hubert_ecg_classification import HuBERTForECGClassification as HuBERTClassification

# ARGS
EPS = 1e-9
DROPOUT_DYNAMIC_REG_FACTOR = 0.05
LR = 1e-5
BETAS = (0.9, 0.98)
WEIGHT_DECAY = 0.01
RAMP_UP_PERC = 0.08
LAYERDROP = 0.0
MODEL_DROPOUT_MULT = -2
FS = 100

class BalancedAccuracy(Metric):
    def __init__(self, num_classes=None, threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        if self.num_classes==1:
            self.recall = BinaryRecall(threshold=threshold)
            self.spec = BinarySpecificity(threshold=threshold)
        else:
            self.recall = MulticlassRecall(num_classes=num_classes, average="macro")

    def update(self, preds, target):
        self.recall.update(preds, target)
        if self.num_classes==1:
            self.spec.update(preds, target)

    def compute(self):
        if self.num_classes==1:
            return (self.recall.compute() + self.spec.compute()) / 2
        else:
            return self.recall.compute()

    def reset(self):
        self.recall.reset()
        if self.num_classes==1:
            self.spec.reset()

def task2metrics(vocab_size):
    if vocab_size == 1:
        return {
            'accuracy' : BinaryAccuracy(),
            'bacc' : BalancedAccuracy(num_classes=1),
            'auroc' : BinaryAUROC(),
            'auprc' : BinaryAUPRC()
        }
    else:
        return {
            'accuracy' : MulticlassAccuracy(num_classes=vocab_size, average="micro"),
            'bacc' : BalancedAccuracy(num_classes=vocab_size),
            'auroc' : MulticlassAUROC(num_classes=vocab_size),
            'auprc' : MulticlassAUPRC(num_classes=vocab_size),
        }

class RemapUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "HuBERTECGConfig":
            return getattr(sys.modules[__name__], name)
        return super().find_class(module, name)        

class RemapPickleModule:
    Unpickler = RemapUnpickler
    load  = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump  = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)

class HECGDataset(Dataset):
    def __init__(self, dataset, n_cls):
        self.ecg = np.array([x for x,_ in dataset])
        self.N = len(self.ecg)
        self.ecg = self.ecg[:,:, :5 * FS] # take first 5s
        self.ecg = self.ecg.reshape(self.N,-1) # flatten
        
        self.labels = np.array([int(y) for _,y in dataset])
        self.labels = np.eye(n_cls)[self.labels]
        self.weights = self._compute_pos_weights()
        print(f"Class Weights: {self.weights}")

    def _compute_pos_weights(self):
        counts = self.labels.sum(axis=0)
        weights = 1.0 / np.sqrt(counts + 1e-9)
        # weights = (self.N - counts) / (counts + 1e-9)
        # weights = weights/weights.mean()

        return torch.FloatTensor(weights)

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, idx):
        ecg = self.ecg[idx]
        max_ecg = np.max(np.abs(ecg), axis=-1, keepdims=True)
        ecg = ecg / (max_ecg + 1e-12)
        
        attn_mask = self._compute_attention_mask(ecg)
        labels = self.labels[idx]
        return (
                torch.from_numpy(ecg).float(),
                torch.from_numpy(attn_mask).long(),
                torch.from_numpy(labels).float()
            )

    def _compute_attention_mask(self, array):
        array = array.reshape(12, -1)     # 12 x SAMPLES_IN_5_SECONDS_AT_500HZ   
        for index in range(array.shape[1]):
            if np.any(array[:, index]):
                break
        start = index
        for index in range(array.shape[1]-1, -1, -1):
            if np.any(array[:, index]):
                break
        end = index
        attention_mask = np.zeros(array.shape)
        attention_mask[:,start:end+1] = 1
        attention_mask = np.concatenate(attention_mask, axis=0)
        return attention_mask

    def collate(self, batch : Tuple[Any]):
        unpacked = tuple(zip(*batch))
        return tuple(map(torch.stack, unpacked))


class HuBERTECGModule():
    def __init__(self, n_outputs, ckpt_path, train_head_only=False):
        self.n_labels = n_outputs
        self.train_head_only = train_head_only

        if ckpt_path is None:
            ckpt_path = "models/HuBERTECG/ckpt/hubert_ecg_small.pt"
        self._load_model(ckpt_path)

        self.results = {}
        for metric in task2metrics(2).keys():
            self.results[f'train_{metric}'] = []
            self.results[f'val_{metric}'] = []

    def _load_model(self, ckpt_path):
        checkpoint = torch.load(
            ckpt_path,
            map_location='cpu',
            weights_only=False,
            pickle_module=RemapPickleModule
        )

        config = checkpoint['model_config']
        
        config.layerdrop = LAYERDROP
        
        pretrained_hubert = HuBERT(config)
        pretrained_hubert.load_state_dict(checkpoint['model_state_dict'])

        for name, module in pretrained_hubert.named_modules():
            if 'dropout' in name:
                module.p = 0.1 + DROPOUT_DYNAMIC_REG_FACTOR * MODEL_DROPOUT_MULT

        self.model = HuBERTClassification(
            pretrained_hubert, 
            num_labels=self.n_labels
            )

        if self.train_head_only:
            for n, p in pretrained_hubert.named_parameters():
                p.requires_grad = ('classifier' in n)
        else:
            self.model.set_transformer_blocks_trainable(n_blocks=config.num_hidden_layers)
            self.model.set_feature_extractor_trainable(True)

    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        device = torch.device('cuda')

        self.model.to(device)

        ### DATA SETUP ###
        train_dataset = HECGDataset(train_dataset, self.n_labels)
        train_dl = DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate,
            num_workers=6,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False
        )
        train_pos_weights = train_dataset.weights

        validation_dataset = HECGDataset(validation_dataset, self.n_labels)
        val_dl = DataLoader(
            validation_dataset,
            collate_fn=validation_dataset.collate,
            num_workers=6,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

        ### OPTIMISATION SETUP ###
        if self.n_labels == 1:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_pos_weights).to(device)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=train_pos_weights).to(device)

        parameters_group = [{"params" : filter(lambda p : p.requires_grad, self.model.parameters()), "lr": LR}]
        optimizer = optim.AdamW(
            parameters_group,
            betas=BETAS,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
        )
        
        training_steps = len(train_dl) * epochs
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=ceil(RAMP_UP_PERC * training_steps),
            num_training_steps=training_steps,
        )
        
        scaler = amp.GradScaler('cuda')
        
        metrics = task2metrics(self.n_labels)
        for name, metric in metrics.items():
            metric.to(device)
       
        ### FINETUNING ###
        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs-1}")
            self.model.train()

            ### training ###
            for name, metric in metrics.items():
                metric.reset()
            for ecg, attention_mask, labels in tqdm(train_dl, total=len(train_dl)):
                ecg = ecg.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.squeeze().to(device)
                with amp.autocast('cuda'):
                    logits, _ = self.model(ecg, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False, return_dict=False)
                    logits = logits.squeeze()
                    loss = criterion(logits, labels)
                    
                scaler.scale(loss).backward() # accumulate normalized loss           
                    
                scaler.step(optimizer)
                lr_scheduler.step()
                scaler.update()
                optimizer.zero_grad()

                if self.n_labels > 1:
                    labels = labels.argmax(dim=1)
                for name, metric in metrics.items():
                    metric.update(logits, labels.long())
            for name, metric in metrics.items():
                score = metric.compute()
                macro = score.mean()
                self.results[f'train_{name}'].append(macro.cpu())
                print(f"Training {name} = {macro}")
            
            ### validation ###                        
            self.model.eval()        
            confusion_matrix = MulticlassConfusionMatrix(num_classes=self.n_labels).to(device)
            for name, metric in metrics.items():
                metric.reset()

            for ecg, _, labels in tqdm(val_dl, total=len(val_dl)):
                ecg = ecg.to(device)
                labels = labels.squeeze().to(device)
                
                with torch.no_grad():
                    logits, _ = self.model(ecg, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False)
                    logits = logits.squeeze()
                
                confusion_matrix.update(logits.argmax(dim=1), labels.argmax(dim=1))

                if self.n_labels > 1:
                    labels = labels.argmax(dim=1)
                for name, metric in metrics.items():
                    metric.update(logits, labels.long())

            torch.set_printoptions(threshold=float("inf"), linewidth=200)
            print(confusion_matrix.compute())
            for name, metric in metrics.items():
                score = metric.compute()
                macro = score.mean()
                self.results[f'val_{name}'].append(macro.cpu())
                print(f"Validation {name} = {macro}")                

