# ECG-FM

The ECG-FM model requires an older version of python (<= 3.9.0) and may therefore be incompatible with the libraries used by other models. It is recommended to create a new environment specifically for benchmarking ECG-FM.

## Environment Setup
[![python](https://img.shields.io/badge/Python-3.9.0-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

### Step 1
 Install [PyTorch](https://pytorch.org/get-started/locally/).

### Step 2
This model is dependent on the [fairseq-signals](https://github.com/Jwoo5/fairseq-signals/) library. First, clone the repository to a local directory `fairseq-signals/`:

```commandline
git clone https://github.com/Jwoo5/fairseq-signals.git
```

Then install to your environment with pip:
```commandline
pip install -e fairseq-signals/
```

### Step 3
Install other requirements:

```commandline
pip install -r requirements.txt
```