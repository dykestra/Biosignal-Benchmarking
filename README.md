# Biosignal Benchmarking

A unified codebase for fine-tuning and evaluation of biosignal models.

## Models

Benchmarking is currently supported for the following models:

### EEG Models

- **EEGNet:** [braindecode](https://braindecode.org/stable/generated/braindecode.models.EEGNetv1.html)
- **EEGInception:** [braindecode](https://braindecode.org/stable/generated/braindecode.models.EEGInception.html)
- **LaBraM:** [github](https://github.com/935963004/LaBraM)
- **NeuroGPT:** [github](https://github.com/wenhui0206/NeuroGPT)
- **CBraMod:** [github](https://github.com/wjq-learning/CBraMod)
- **BIOT:** [github](https://github.com/ycq091044/BIOT)
- **EEGPT:** [github](https://github.com/BINE022/EEGPT)
- **MIRepNet:** [github](https://github.com/staraink/MIRepNet)

### ECG Models

- **HuBERT-ECG:** [github](https://github.com/Edoar-do/HuBERT-ECG)
- **ECGFounder:** [github](https://github.com/PKUDigitalHealth/ECGFounder)
- **ECG-FM:** [github](https://github.com/bowang-lab/ECG-FM/)

## Environment Setup
[![python](https://img.shields.io/badge/Python-3.11.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

Install [PyTorch](https://pytorch.org/get-started/locally/).

Install other requirements:

```commandline
pip install -r requirements.txt
```

## EEG Data Pre-processing

### Download EEG Data

The following 5 EEG datasets are selected for benchmarking:
| Dataset | Paradigm | Number of Classes | Type(s) | Tasks |
|-|-|-|-|-|
| [High Gamma](https://github.com/robintibor/high-gamma-dataset) | Executed Movement | 4 | | `no_action`, `left_fist`, `right_fist`, `both_feet` |
| [OpenBMI-ERP](http://gigadb.org/dataset/100542) | ERP | 2 | | `target`, `nontarget` |
| [Pavlov 2022](https://openneuro.org/datasets/ds003838/versions/1.0.2) | Working Memory | 2 | `13_digits` | `memory`, `control` |
| [Sleep-EDF](https://www.physionet.org/content/sleep-edfx/1.0.0/) | Sleep Stage | 6 | | `Sleep stage W`, `Sleep stage 1`, `Sleep stage 2`, `Sleep stage 3`, `Sleep stage 4`, `Sleep stage R` |
| [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) | Eyes Open-Closed | 2 | `eye_open`, `eye_closed` | |

### Pre-process EEG Signal Data
The following pre-processing should be applied to the raw EEG signals for each dataset:
- resampled to sample frequency for the selected model
- bandpass filter as needed for the selected model
- notch filtered at 50Hz, 60Hz and harmonics (if not already excluded with bandpass)
- cut into trials:
  - **High Gamma:** 0s-4s after each cue
  - **OpenBMI-ERP:** 0.2s before - 0.8s after each cue
  - **Pavlov 2022:** 14s-18s after each 13 digits trial cue (corresponding to the peak in pupil size reported in the dataset publication)
  - **Sleep-EDF:** 30s epochs from the original continuous recording, and discard any data from the awake condition except for the 30 minutes before and after sleep
  - **PhysioNet:** 4s epochs from the original continuous recording (only runs 1 and 2)
- saved in numpy format as an array with shape (N_trials, Channels, Time)

### Save Metadata
For compatability with our data loading functions, metadata about each dataset should be saved as a pandas DataFrame where each row corresponds to a single trial
- each row should contain the subject ID
- each row should give the 'task' or 'type' of the trial
- the attributes of the file should include the list of channel names

## ECG Data Pre-processing
The [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3) dataset is currently used for ECG benchmarking, with labels for 5, 23 or 43 classes.

### Pre-process ECG Signal Data


## Run Benchmarking
Once data has been pre-processed to the expected format and saved under `{DATA_PATH}`, the benchmarking script can be run as below:

```commandline
python main.py \
  --data-root={DATA_PATH} \
  --batch-size=64 \
  --n-epochs=100 \
  --model-name='EEGNetv1' \
  --output-dir='results'
```

Alternatively the bash script `finetune.sh` can be run which has our minimal configuration and can also take the experiment ID number for ClearML as an argument e.g.
```commandline
source finetune.sh 123
```
By default this will run training/fine-tuning on the whole model, unless the `--train_head_only` argument is added to freeze the pre-trained model.

### Mode Selection
By default, the `mode` is set to `'benchmark'` which will run cross-fold validation and return metrics. 

To run training on the whole dataset and save the finetuned model, set `--mode='finetune'` in the bash script.

## Citation
```
@article{Lee_2025,
title={A Comprehensive Review of Biosignal Foundation Models},
url={http://dx.doi.org/10.36227/techrxiv.176369849.97173246/v1},
DOI={10.36227/techrxiv.176369849.97173246/v1},
publisher={Institute of Electrical and Electronics Engineers (IEEE)},
author={Lee, Na and Barmpas, Konstantinos and Koliousis, Alexandros and Panagakis, Yannis and Adamos, Dimitrios and Laskaris, Nikolaos and Zafeiriou, Stefanos},
year={2025},
month=nov }
```

```
@inproceedings{
lee2025assessing,
title={Assessing the Capabilities of Large Brainwave Foundation Models},
author={Na Lee and Stylianos Bakas and Konstantinos Barmpas and Yannis Panagakis and Dimitrios Adamos and Nikolaos Laskaris and Stefanos Zafeiriou},
booktitle={IEEE International Workshop on Machine Learning for Signal Processing (MLSP) 2025, Special Sessions},
year={2025},
url={https://openreview.net/forum?id=Qwn2a1uIpx}
}
```
