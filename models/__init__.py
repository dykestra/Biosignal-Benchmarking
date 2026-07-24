from importlib import import_module

__all__ = [
    "BIOTModule",
    "CBraModModule",
    "EEGPTModule",
    "LaBraMModule",
    "NeuroGPTModule",
    "MIRepNetModule",
    "HuBERTECGModule",
    "ECGFounderModule",
    "ECG_FMModule",
    "LUNAModule",
    "REVEModule",
    "NeuroRVQModule"
]

_MODULE_PATHS = {
    "BIOTModule": "models.BIOT.modules",
    "CBraModModule": "models.CBraMod.modules",
    "EEGPTModule": "models.EEGPT.modules",
    "LaBraMModule": "models.LaBraM.run_class_finetuning",
    "NeuroGPTModule": "models.NeuroGPT.modules",
    "MIRepNetModule": "models.MIRepNet.modules",
    "HuBERTECGModule": "models.HuBERTECG.modules",
    "ECGFounderModule": "models.ECGFounder.modules",
    "ECG_FMModule": "models.ECG_FM.modules",
    "LUNAModule": "models.LUNA.modules",
    "REVEModule": "models.REVE.modules",
    "NeuroRVQModule": "models.NeuroRVQm.modules"
}

def __getattr__(name):
    if name not in _MODULE_PATHS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = import_module(_MODULE_PATHS[name])
    obj = getattr(mod, name)

    # cache it so next access is fast
    globals()[name] = obj
    return obj