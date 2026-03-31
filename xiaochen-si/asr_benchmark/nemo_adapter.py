"""NeMo adapter config utilities.

Helper functions for setting up NeMo ASR adapter training. Extracted from the vendored 
https://github.com/NVIDIA-NeMo/NeMo/blob/86264e7c45f6fa2d30045c90e751637bfcb79d1c/examples/asr/asr_adapters/train_asr_adapter.py script so they can be
reused from notebooks without Hydra.
"""

import os
from dataclasses import is_dataclass
from typing import Union

from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.core import adapter_mixins


def update_model_config_to_support_adapter(model_cfg, current_cfg):
    """Swap the encoder target class to its adapter-compatible variant.

    Also copies the ``log_prediction`` flag from *current_cfg* into
    *model_cfg* so the adapter training loop can honour it.
    """
    with open_dict(model_cfg):
        model_cfg.log_prediction = current_cfg.model.get('log_prediction', False)

        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if adapter_metadata is not None:
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path


def update_model_cfg(original_cfg, new_cfg):
    """Merge a user-supplied dataset config into the model's default config.

    Keys in *new_cfg* that are on the whitelist (``num_workers``,
    ``pin_memory``, ``batch_size``, ``use_lhotse``, ``channel_selector``)
    are always injected.  Other keys that don't already exist in
    *original_cfg* are silently dropped.
    """
    with open_dict(original_cfg), open_dict(new_cfg):
        whitelist_keys = ['num_workers', 'pin_memory', 'batch_size', 'use_lhotse', 'channel_selector']
        for wkey in whitelist_keys:
            if wkey in new_cfg:
                original_cfg[wkey] = new_cfg[wkey]
                print(f"Injecting white listed key `{wkey}` into config")

        new_keys = list(new_cfg.keys())
        for key in new_keys:
            if key not in original_cfg:
                new_cfg.pop(key)
                print("Removing unavailable key from config :", key)

        new_cfg = OmegaConf.merge(original_cfg, new_cfg)
    return new_cfg


def add_global_adapter_cfg(model: ASRModel, global_adapter_cfg: Union[DictConfig, dict]):
    """Attach a global adapter config to the model.

    Converts *global_adapter_cfg* to a ``DictConfig`` if needed, then
    stores it under ``model.cfg.adapters`` and propagates it to every
    adapter module that already exists on the model.
    """
    if is_dataclass(global_adapter_cfg):
        global_adapter_cfg = OmegaConf.structured(global_adapter_cfg)

    if not isinstance(global_adapter_cfg, DictConfig):
        global_adapter_cfg = DictConfig(global_adapter_cfg)

    with open_dict(global_adapter_cfg), open_dict(model.cfg):
        if 'adapters' not in model.cfg:
            model.cfg.adapters = OmegaConf.create({})

        model.cfg.adapters[model.adapter_global_cfg_key] = global_adapter_cfg
        model.update_adapter_cfg(model.cfg.adapters)


def patch_transcribe_lhotse(model: ASRModel) -> None:
    """Monkey-patch ``_setup_transcribe_dataloader`` to disable lhotse.

    NeMo's ``EncDecRNNTBPEModel._setup_transcribe_dataloader`` hardcodes
    ``use_lhotse=True``.  When the installed lhotse version is incompatible
    with the current PyTorch ``Sampler`` API this causes a ``TypeError`` at
    inference time.  This patch replaces the method on *model* so that
    ``use_lhotse`` is ``False``.
    """

    def _patched(config):
        if "manifest_filepath" in config:
            manifest_filepath = config["manifest_filepath"]
            batch_size = config["batch_size"]
        else:
            manifest_filepath = os.path.join(config["temp_dir"], "manifest.json")
            batch_size = min(config["batch_size"], len(config["paths2audio_files"]))

        dl_config = {
            "use_lhotse": False,
            "manifest_filepath": manifest_filepath,
            "sample_rate": model.preprocessor._sample_rate,
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": config.get("num_workers", min(batch_size, os.cpu_count() - 1)),
            "pin_memory": True,
            "channel_selector": config.get("channel_selector", None),
            "use_start_end_token": model.cfg.validation_ds.get("use_start_end_token", False),
        }
        if config.get("augmentor"):
            dl_config["augmentor"] = config["augmentor"]
        return model._setup_dataloader_from_config(config=DictConfig(dl_config))

    model._setup_transcribe_dataloader = _patched
