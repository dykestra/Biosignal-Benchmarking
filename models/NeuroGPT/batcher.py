#!/usr/bin/env python3
import pdb
from typing import Dict
import numpy as np
import torch
import h5py
import os

from torch.utils.data import Dataset

def _pad_seq_right_to_n(
    seq: np.ndarray,
    n: int,
    pad_value: float = 0.
    ) -> np.ndarray:
    if n == seq.shape[0]:
        return seq
    return np.concatenate(
        [
            seq,
            np.ones(
                (
                    n-seq.shape[0],
                    *seq.shape[1:]
                )
            ) * pad_value,  
        ],
        axis=0,
    )

REORDER_LABELS = {'FP1': 0, 'FP2': 1, 'F7': 2, 'F3': 3, 'FZ': 4, 'F4': 5, 'F8': 6, 'T1': 7, 'T3': 8, 'C3': 9,
         'CZ': 10, 'C4': 11, 'T4': 12, 'T2': 13, 'T5': 14, 'P3': 15, 'PZ': 16, 'P4': 17, 'T6': 18, 'O1': 19, 'OZ': 20, 'O2': 21}

class EEGDataset(Dataset):
    def __init__(self, eeg, labels, chnames, sample_keys=['inputs', 'attention_mask'], chunk_len=500, num_chunks=2, ovlp=0, root_path="", population_mean=0, population_std=1, gpt_only=False, normalization=True, start_samp_pnt=-1):
        self.eeg = eeg
        self.labels = labels
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.ovlp = ovlp
        self.sample_keys = sample_keys
        self.mean = population_mean
        self.std = population_std
        self.do_normalization = normalization
        self.gpt_only=gpt_only
        self.start_samp_pnt = start_samp_pnt
        self.chann_labels = {}
        for i, c in enumerate(chnames):
            self.chann_labels[c] = i

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        data = self.eeg[idx]
        if len(data) < self.chunk_len:
            pad = (self.chunk_len - len(data)) // 2
            data = np.pad(data, ((0,0),(pad, pad)), 'constant')
            self.ovlp = self.chunk_len
        data = self.reorder_channels(data)
        return self.preprocess_sample(data, self.num_chunks, self.labels[idx])

    @staticmethod
    def _pad_seq_right_to_n(
        seq: np.ndarray,
        n: int,
        pad_value: float = 0
        ) -> np.ndarray:
        return _pad_seq_right_to_n(
            seq=seq,
            n=n,
            pad_value=pad_value
        )

    def reorder_channels(self, data):
        reordered = np.zeros((len(REORDER_LABELS), data.shape[-1]))
        for label, target_idx in REORDER_LABELS.items():
            if label not in self.chann_labels.keys():
                # 10-20 to 10-10 equivalent
                if label == "T3" and "T7" in self.chann_labels:
                    label = "T7"
                elif label == "T4" and "T8" in self.chann_labels:
                    label = "T8"
                elif label == "T5" and "P7" in self.chann_labels:
                    label = "P7"
                elif label == "T6" and "P8" in self.chann_labels:
                    label = "P8"
                elif label == "T1":
                    if "FT9" in self.chann_labels:
                        label = "FT9"
                    elif "FT7" in self.chann_labels:
                        label = "FT7"
                elif label == "T2":
                    if "FT10" in self.chann_labels:
                        label = "FT10"
                    elif "FT8" in self.chann_labels:
                        label = "FT8"

            if label in self.chann_labels.keys():
                mapped_idx = self.chann_labels[label]
                reordered[target_idx, :] = data[mapped_idx, :]
        return reordered

    def split_chunks(self, data, length=500, ovlp=50, num_chunks=10, start_point=-1): 
        '''2 seconds, 0.2 seconds overlap'''
        all_chunks = []
        total_len = data.shape[1]
        actual_num_chunks = num_chunks
        
        if start_point == -1:
            if length == ovlp:
                # for cases where chunk needs to be repeated to make num_chunks
                start_point = 0
            elif num_chunks * length > total_len - 1:
                start_point = 0
                # actual_num_chunks = total_len // length
                actual_num_chunks = int(np.floor((total_len - length) / (length - ovlp)) + 1)
            else:
                start_point = np.random.randint(0, total_len - num_chunks * length)
        
        for i in range(actual_num_chunks):
            chunk = data[:, start_point: start_point + length]
            all_chunks.append(np.array(chunk))
            start_point = start_point + length - ovlp
        return np.array(all_chunks), start_point
    
    def normalize(self, data):
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        # Ensure std is not zero to avoid division by zero.
        # If std is zero, normalization doesn't make sense, 
        # so you might set std to a small positive value or handle it in another way.
        # std = np.where(std == 0, 1e-23, std)
        return (data - mean) / (std + 1e-25)

    def preprocess_sample(
        self,
        sample,
        seq_len,
        labels=None
        ) -> Dict[str, torch.Tensor]:
        # pdb.set_trace()
        out = {}
        if self.do_normalization:
            sample = self.normalize(sample)

        chunks, seq_on = self.split_chunks(sample, self.chunk_len, self.ovlp, seq_len, self.start_samp_pnt)

        attention_mask = np.ones(seq_len)
        chunks = self._pad_seq_right_to_n(
            seq=chunks,
            n=seq_len,
            pad_value=0
        )

        attention_mask = self._pad_seq_right_to_n(
            seq=attention_mask, 
            n=seq_len,
            pad_value=0
        )
        
        if self.gpt_only == True:
            chunks = np.reshape(chunks, (seq_len, chunks.shape[1]*chunks.shape[2]))
        out["inputs"] = torch.from_numpy(chunks).to(torch.float)
        out["attention_mask"] = torch.from_numpy(attention_mask).to(torch.long)
        out['seq_on'] = seq_on
        out['seq_len'] = seq_len
        
        if self.sample_keys is not None:
            out = {
                key: out[key] 
                for key in self.sample_keys
                if key in out
            }

        if labels is not None:
            out['labels'] = torch.from_numpy(np.array(labels)).to(torch.long)
   
        return out