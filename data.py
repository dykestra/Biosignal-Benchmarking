import pdb
import os
import numpy as np
import pandas as pd
from abc import ABC
import pickle

def common_average_reference(eeg):
    # Apply common average referencing to signal eeg: (N, C, T)
    return eeg - eeg.mean(axis=-2, keepdims=True)

def get_data_dir(root, main_dir, subdir=None):
    """
    Returns: Path [root]/[main_dir] or [root]/[main_dir]/[subdir] if subdir exists
    """
    dir = main_dir if subdir is None else os.path.join(main_dir, subdir)
    return os.path.join(root, dir)


class Benchmark(ABC):
    """
    Class for benchmark dataset with expected properties:
        data: array of biosignal data (samples, channels, time)
        subject_ids: array of subject ID for each data sample (samples,)
        labels: array of target class labels for each data sample (samples,)
        chnames: array of electrode channel names (channels,)
    """
    def __init__(self):
        self.data = None
        self.subject_ids = None
        self.labels = None
        self.chnames = None
    
    def get_data(self):
        return self.data, self.subject_ids, self.labels, self.chnames

    def sample_balanced_set(self, idx, seed):
        """
        Performs a random sampling of indices to balance classes for each subject
            idx: array of sample indices relative to self.data
            seed: random seed for sampling
        Returns:
            filtered indices after random sampling 
        """
        rng = np.random.default_rng(seed)

        subj_all = self.subject_ids[idx]
        y_all = self.labels[idx]

        sampled = []

        for s in np.unique(subj_all):
            mask_s = (subj_all == s)
            idx_s = idx[mask_s]
            y_s = y_all[mask_s]

            labels = np.unique(y_s)

            idx_by_label = [idx_s[y_s == label] for label in labels]

            # minority per subject
            n = min([len(idx_l) for idx_l in idx_by_label])
            if n == 0:
                continue

            take_by_label = [rng.choice(idx_l, size=n, replace=False) for idx_l in idx_by_label]
            sampled.append(np.concatenate(take_by_label))

        sampled_idx = np.concatenate(sampled)
        return sampled_idx
    
    def get_splits(self, n_splits):
        print("No predefined splits found for this benchmark")


class KUERPBenchmark(Benchmark):
    def __init__(self, root, subdir, apply_car, **kwargs):
        print("Loading KU ERP...")
        dir = get_data_dir(root, 'KU_ERP', subdir)
        kuerp_eeg = np.load(os.path.join(dir, 'kuerp_data.npy'), mmap_mode='r')
        kuerp_tf = pd.read_pickle(os.path.join(dir, 'kuerp_trial_features.pd'))

        trial_mask = (kuerp_tf['task'] == 'target') | (kuerp_tf['task'] == 'nontarget')
        kuerp_tf = kuerp_tf[trial_mask]
        kuerp_eeg = kuerp_eeg[trial_mask, :, :]

        if apply_car:
            kuerp_eeg = common_average_reference(kuerp_eeg)

        labels = kuerp_tf['task'].replace({'nontarget': 0, 'target': 1}).to_numpy()
        subject_ids = kuerp_tf['subject_id'].to_numpy()

        chnames = np.array([c.upper() for c in kuerp_tf.attrs['channel_names']])
        
        self.data = kuerp_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames


class PhysionetMIBenchmark(Benchmark):
    def __init__(self, root, subdir=None, apply_car=False, **kwargs):
        super().__init__()
        print("Loading PhysionetMI...")
        dir = get_data_dir(root, "PhysionetMI", subdir)
        physioeyes_eeg = np.load(os.path.join(dir, 'mmidb_data.npy'), mmap_mode='r')
        physioeyes_tf = pd.read_pickle(os.path.join(dir, 'mmidb_trial_features.pd'))

        sample_rate = physioeyes_tf['target_fs'][0]

        trial_mask = (physioeyes_tf['type'] == 'eye_open') | (physioeyes_tf['type'] == 'eye_closed')
        physioeyes_tf = physioeyes_tf[trial_mask]

        physioeyes_eeg = physioeyes_eeg[trial_mask, :, :int(4 * sample_rate)]  # cut for max n_patches

        if apply_car:
            physioeyes_eeg = common_average_reference(physioeyes_eeg)

        labels = physioeyes_tf['type'].replace({'eye_closed': 0, 'eye_open': 1}).to_numpy()
        subject_ids = physioeyes_tf['subject_id'].to_numpy()

        chnames = np.array([c.upper() for c in physioeyes_tf.attrs['channel_names']])

        self.data = physioeyes_eeg
        self.subject_ids = subject_ids
        self.labels =labels
        self.chnames = chnames

    def sample_balanced_set(self, idx, seed):
        print("Classes are already balanced for Physionet MI")
        return idx


class Pavlov22Benchmark(Benchmark):
    def __init__(self, root, subdir=None, apply_car=False, **kwargs):
        super().__init__()
        print("Loading Pavlov22...")
        dir = get_data_dir(root, "Pavlov22", subdir)
        pavlov_eeg = np.load(os.path.join(dir, 'pavlov2022_data.npy'), mmap_mode='r')
        pavlov_tf = pd.read_pickle(os.path.join(dir, 'pavlov2022_trial_features.pd'))

        sample_rate = pavlov_tf['target_fs'][0]
        trial_mask = (pavlov_tf['task'] == 'memory') | (pavlov_tf['task'] == 'control')
        trial_mask = trial_mask & (pavlov_tf['type'] == '13_digits')
        pavlov_tf = pavlov_tf[trial_mask]
        pavlov_eeg = pavlov_eeg[trial_mask, :, int(18. * sample_rate):int(22. * sample_rate)]

        if apply_car:
            pavlov_eeg = common_average_reference(pavlov_eeg)

        labels = pavlov_tf['task'].replace({'control': 0, 'memory': 1}).to_numpy()
        subject_ids = pavlov_tf['subject_id'].to_numpy()

        chnames = np.array([c.upper() for c in pavlov_tf.attrs['channel_names']])

        self.data = pavlov_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames


class SleepEDFBenchmark(Benchmark):
    def __init__(self, root, subdir, apply_car, **kwargs):
        print("Loading Sleep EDF...")
        dir = get_data_dir(root, "SleepEDF", subdir)
        sleep_eeg = np.load(os.path.join(dir, 'SleepEDF_eeg_trials.npy'), mmap_mode='r')
        sleep_tf = pd.read_pickle(os.path.join(dir, 'SleepEDF_trial_features.pd'))

        if apply_car:
            sleep_eeg = common_average_reference(sleep_eeg)

        labels = sleep_tf['task'].to_numpy()
        subject_ids = sleep_tf['subject_id'].to_numpy()

        chnames = np.array([c.split('-')[0].upper() for c in sleep_tf.attrs['channel_names']])
        
        self.data = sleep_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames
    

class HighGammaBenchmark(Benchmark):
    def __init__(self, root, subdir, apply_car, n_cls):
        super().__init__()
        print("Loading High Gamma...")
        dir = get_data_dir(root, "HighGamma", subdir)
        hgd_eeg = np.load(os.path.join(dir, 'highgamma_data.npy'), mmap_mode='r')
        hgd_tf = pd.read_pickle(os.path.join(dir, 'highgamma_trial_features.pd'))

        # mask out channels
        hgd_chnames = hgd_tf.attrs['channel_names']
        non_EEG = ['EOGh', 'EOGv', 'EMG_RH', 'EMG_LH', 'EMG_RF']
        other_EEG = ['AFF1', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',
            'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'I2', 
            'AFp3h', 'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h', 
            'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h', 'PPO9h', 'PPO5h', 'PPO6h', 'PPO10h', 'POO9h',
            'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h'] # i.e. electrodes not in standard 10-20
        hgd_chmask = np.invert(np.isin(hgd_chnames, non_EEG + other_EEG))
        hgd_eeg = hgd_eeg[:, hgd_chmask, :]
        hgd_chnames = hgd_chnames[hgd_chmask]

        sample_rate = hgd_tf['target_fs'][0]
        hgd_eeg = hgd_eeg[:, :, int(2.75 * sample_rate):int(6.75 * sample_rate)] # cut out 4s trial

        if apply_car:
            hgd_eeg = common_average_reference(hgd_eeg)

        if n_cls == 2:
            # action vs no-action
            labels = hgd_tf['task'].replace({'no_action': 0, 'left_fist': 1, 'right_fist': 1, 'both_feet': 1}).to_numpy()
        else:
            if n_cls != 4:
                print(f'No label assignment for {n_cls} classes, using default 4 classes')
            labels = hgd_tf['task'].replace({'no_action': 0, 'left_fist': 1, 'right_fist': 2, 'both_feet': 3}).to_numpy()

        subject_ids = hgd_tf['subject_id'].to_numpy()

        chnames = np.array([c.upper() for c in hgd_chnames])
        
        self.data = hgd_eeg
        self.subject_ids = subject_ids
        self.labels = labels
        self.chnames = chnames

    def sample_balanced_set(self, idx, seed):
        print("Classes are already balanced for High Gamma")
        return idx

class PTBXLBenchmark(Benchmark):
    def __init__(self, root, subdir, **kwargs):
        super().__init__()
        print("Loading PTB-XL...")
        assert kwargs['n_cls'] in [5, 23, 43], "Unsupported number of classes"

        self.benchmark_root = os.path.join(root, "PTB_XL")
        tf = pd.read_csv(os.path.join(self.benchmark_root, "ptb_xl_cut_benchmarking_v2.csv"), keep_default_na=False)
        dir = get_data_dir(root, "PTB_XL", subdir)
        ecg = np.load(os.path.join(dir, 'ptb_xl_cut_benchmarking_v2.npy'), mmap_mode='r')

        label_col = f"diagnostic_{kwargs['n_cls']}_classes"
        tf = tf.loc[:, ['patient_id', 'scp_codes', 'strat_fold', label_col]]

        trial_mask = (tf[label_col] != 'None')
        ecg = ecg[trial_mask]
        self.tf = tf[trial_mask]

        scp_codes = np.unique(self.tf[label_col])
        scp_code_mapping = {label: i for i, label in enumerate(scp_codes)}
        print(f"scp_code mapping: {scp_code_mapping}")
        labels = self.tf[label_col].replace(scp_code_mapping)

        self.data = ecg
        self.subject_ids = np.array(self.tf['patient_id'])
        self.labels = np.array(labels)
        self.chnames = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


    def get_splits(self, n_splits):
        n_folds = len(np.unique(self.tf['strat_fold']))
        assert n_folds == n_splits, f"Incompatible number of splits: {n_splits}. Predefined splits found for {n_folds} folds."
        
        split_path = os.path.join(self.benchmark_root, 'ptb-xl-splits.pkl')
        if os.path.exists(split_path):
            print("Loading predefined splits from file...")
            with open(split_path, 'rb') as file:
                return pickle.load(file)

        sbj_id_unique = np.sort(np.unique(self.subject_ids))
        fold_subjects = {}
        for fold in range(n_folds):
            fold_subjects[fold] = []
            for i, sbj in enumerate(sbj_id_unique):
                filtered = self.tf[(self.tf['patient_id']==sbj) & (self.tf['strat_fold']==fold+1)]
                if len(filtered) > 0:
                    fold_subjects[fold].append(i)

        splits = []
        for fold in reversed(range(n_folds)):
            test_subjects = fold_subjects[fold]
            train_subjects = np.concatenate([s for f,s in fold_subjects.items() if f!=fold])
            splits.append((train_subjects, test_subjects))

        with open(split_path, 'wb') as file:
            pickle.dump(splits, file)

        return splits

def load_benchmark(benchmark, root, subdir, apply_car=False, n_cls=2) -> Benchmark:
    BENCHMARK_CLASSES = {
        "High Gamma": HighGammaBenchmark,
        "KU ERP": KUERPBenchmark,
        "Pavlov memory": Pavlov22Benchmark,
        "Sleep EDF": SleepEDFBenchmark,
        "Physionet MI": PhysionetMIBenchmark,
        "PTB-XL": PTBXLBenchmark
    }

    assert (benchmark in BENCHMARK_CLASSES), f"Unsupported benchmark {benchmark}. Make sure load function is added to BENCHMARK_LOADERS."
    
    benchmark_cls = BENCHMARK_CLASSES[benchmark]
    return benchmark_cls(root, subdir, apply_car=apply_car, n_cls=n_cls)