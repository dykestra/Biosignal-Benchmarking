import pdb
import os
import csv
import clearml

from wrappers import *

class CSVLogger():
    def __init__(self, output_dir, ex_id):
        self.log_dir = os.path.join(output_dir, f"{ex_id}_log")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self._files = set()
    
    def report_scalar(self, title, series, value, iteration):
        '''
        Mimics clearml report_scalar() function to log values to CSV file
        '''
        if 'train' in series:
            filepath = os.path.join(self.log_dir, f"{title}_train.csv")
        else:
            filepath = os.path.join(self.log_dir, f"{title}_val.csv")

        write_header = filepath not in self._files

        with open(filepath, mode="a", newline="") as f:
            writer = csv.writer(f)
            if 'MEAN' in title:
                if write_header:
                    writer.writerow(["Series", "Iteration", "Value"])
                    self._files.add(filepath)
                writer.writerow([series, iteration, value])
            else:
                if write_header:
                    writer.writerow(["Fold", "Iteration", "Value"])
                    self._files.add(filepath)
                writer.writerow([series.split(' ')[-1], iteration, value])

def get_logger(args):
    if args.logger == 'clearml':
        task = clearml.Task.init(project_name='BRWD-254 Foundation Models',
                                    task_name='{0:04d}'.format(args.ex_id),
                                    task_type=clearml.TaskTypes.training,
                                    auto_connect_frameworks=False)
        logger = clearml.Logger.current_logger()
    elif args.logger == 'csv':
        logger = CSVLogger(args.output_dir, args.ex_id)
    else:
        print(f"No implementation found for logger: {args.logger}")

    return logger

def get_model(model_name, n_chans, ch_names, sfreq, n_times, n_outputs, sbj_ids, encoder_only=False, ckpt_path=None, modality='eeg', train_head_only=False):
    """
    Returns: FinetuningWrapper for the specified model
    """
    if model_name == "EEGNet":
        return EEGNetv1Wrapper(
            n_chans=n_chans, 
            n_times=n_times, 
            sfreq=sfreq, 
            n_outputs=n_outputs
            )
    elif model_name == "EEGInception":
        return EEGInceptionWrapper(
            n_chans=n_chans, 
            n_times=n_times, 
            sfreq=sfreq, 
            n_outputs=n_outputs
        )
    elif model_name == "LaBraM":
        return LaBraMWrapper(
            ch_names=ch_names,
            sfreq=sfreq,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "EEGPT":
        return EEGPTWrapper(
            ch_names=ch_names,
            n_outputs=n_outputs,
            n_times=n_times,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "NeuroGPT":
        return NeuroGPTWrapper(
            ch_names=ch_names,
            n_outputs=n_outputs,
            encoder_only=encoder_only,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "CBraMod":
        return CBraModWrapper(
            ch_names=ch_names,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "BIOT":
        return BIOTWrapper(
            ch_names=ch_names,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "MIRepNet":
        return MIRepNetWrapper(
            ch_names=ch_names,
            n_outputs=n_outputs,
            sbj_ids=sbj_ids,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "LUNA":
        return LUNAWrapper(
            ch_names=ch_names,
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "REVE":
        return REVEWrapper(
            ch_names=ch_names, 
            sfreq=sfreq, 
            n_outputs=n_outputs, 
            n_time=n_times, 
            ckpt_path=ckpt_path, 
            train_head_only=train_head_only
        )
    elif model_name == "HuBERTECG":
        return HuBERTECGWrapper(
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "ECGFounder":
        return ECGFounderWrapper(
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "ECG-FM":
        return ECG_FMWrapper(
            n_outputs=n_outputs,
            ckpt_path=ckpt_path,
            train_head_only=train_head_only
        )
    elif model_name == "NeuroRVQ":
        return NeuroRVQWrapper(
            n_time=n_times, 
            ch_names=ch_names, 
            n_outputs=n_outputs, 
            ckpt_path=ckpt_path, 
            modality=modality, 
            train_head_only=train_head_only
        )
    else:
        print(f"Undefined model name: {model_name}")

def get_subdir(model_name):
    """
    Returns: str name of subdirectory of preprocessed data for given model, 
    should return None if data is stored at root dir
    """
    if model_name in ["EEGNet", "EEGInception", "LaBraM", "CBraMod", "BIOT"]:
        return None
    elif model_name == "EEGPT":
        return "01_100Hz"
    elif model_name in ["NeuroGPT", "MIRepNet"]:
        return "neurogpt_cut"
    elif model_name == "HuBERTECG":
        return "hubertecg"
    elif model_name == "ECGFounder":
        return "ecgfounder"
    elif model_name == "LUNA":
        return "luna"
    else:
        print(f"No specified data subdirectory for model {model_name}")
        return None