
# NOTE: C++–based imports must stay at top—do not move them.
from rdkit import Chem
try:
    import graph_tool  # noqa: F401  # keep optional import for legacy ckpts
except ModuleNotFoundError:
    pass

import os
import pathlib
import warnings
import random
import string
import numpy as np
import torch
import wandb
import hydra
import omegaconf
import sys
from omegaconf import DictConfig, OmegaConf, open_dict
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.warnings import PossibleUserWarning

import utils
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion.extra_features_molecular import ExtraMolecularFeatures
from model.diffusion_discrete import DiscreteDenoisingDiffusion
from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from analysis.visualization import MolecularVisualization
from dataset.ppo_dataset import PPODataModule  # noqa: F401  (kept for RL)

warnings.filterwarnings("ignore", category=PossibleUserWarning)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def init(cfg):
    """Fix random seeds and deterministic flags."""
    seed = cfg.general.seed
    device = cfg.general.device  # Not used directly but kept for completeness

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def generate_random_letter(k: int) -> str:
    letters = string.ascii_letters
    return "".join(random.choices(letters, k=k))


def get_resume(cfg, model_kwargs):
    """Resume in **test‑only** mode (no weight updates)."""
    saved_cfg = cfg.copy()
    name = cfg.general.name + "_resume"
    resume = cfg.general.test_only

    if str(resume).endswith(".pt"):
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        state_dict = torch.load(resume, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)

    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """Resume **training** from an existing checkpoint while allowing hyper‑param changes."""
    saved_cfg = cfg.copy()

    # Resolve absolute path to checkpoint
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split("outputs")[0]
    resume_path = os.path.join(root_dir, cfg.general.resume)

    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, cfg=cfg, **model_kwargs)

    # Merge old + new cfg (give priority to newly supplied keys)
    new_cfg = model.cfg
    for category in cfg:
        for arg in cfg[category]:
            if arg not in new_cfg[category]:
                with open_dict(new_cfg[category]):
                    new_cfg[category][arg] = cfg[category][arg]
            else:
                new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    # Append dataset‑specific identifier to run name
    if cfg.dataset.name in ["zinc", "moses", "chembl", "guacamol"]:
        new_cfg.general.name += f"_resume{cfg.general.target_prop}"
    else:
        new_cfg.general.name += f"_resume{cfg.dataset.name}"

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    print(f"learning rate: {cfg.train.lr}")
    return new_cfg, model


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {
        "name": cfg.general.name,
        "project": f"graph_ddm_{cfg.dataset.name}",
        "config": config_dict,
        "settings": wandb.Settings(_disable_stats=True),
        "reinit": True,
        "mode": cfg.general.wandb,
    }
    wandb.init(**kwargs)
    wandb.save("*.txt")
    return cfg

# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.1", config_path="./configs", config_name="config")
def main(cfg: DictConfig):  # noqa: C901  (function is long but kept for clarity)
    # ---------------------------------------------------------------------
    # Initialisation & dataset selection
    # ---------------------------------------------------------------------
    init(cfg)

    workdir = os.getcwd()
    print("os working dir", workdir)
    home_prefix = "./../../../../" if "multirun" in workdir else "./../../../"

    dataset_config = cfg["dataset"]
    print(dataset_config)

    # ------------------------------------------------------------------
    # Molecular datasets branch (ZINC / MOSES / CHEMBL / GUACAMOL)
    # ------------------------------------------------------------------
    if dataset_config["name"] in ["moses", "zinc", "chembl", "guacamol"]:
        # Datasets & infos ------------------------------------------------
        if dataset_config.name == "zinc":
            from dataset import zinc_dataset as ds
            datamodule = ds.MosesDataModule(cfg)
            dataset_infos = ds.MOSESinfos(datamodule, cfg)
            train_path = os.path.join(home_prefix, "dataset/zinc/raw/zinc_train.csv")
            train_smiles = pd.read_csv(train_path)["smiles"].tolist()
        elif dataset_config.name == "moses":
            from dataset import moses_dataset as ds
            datamodule = ds.MosesDataModule(cfg)
            dataset_infos = ds.MOSESinfos(datamodule, cfg)
            train_path = os.path.join(home_prefix, "dataset/moses/moses_pyg/raw/train_moses.csv")
            train_smiles = pd.read_csv(train_path)["SMILES"].tolist()
        elif dataset_config.name == "chembl":
            from dataset import chembl_dataset as ds
            datamodule = ds.MosesDataModule(cfg)
            dataset_infos = ds.MOSESinfos(datamodule, cfg)
            train_path = os.path.join(home_prefix, "dataset/chembl/raw/chemblv35_train.csv")
            train_smiles = pd.read_csv(train_path)["smiles"].tolist()
        elif dataset_config.name == "guacamol":
            from dataset import guacamol_dataset as ds
            datamodule = ds.MosesDataModule(cfg)
            dataset_infos = ds.MOSESinfos(datamodule, cfg)
            train_path = os.path.join(home_prefix, "dataset/guacamol/raw/guacamol_train.csv")
            train_smiles = pd.read_csv(train_path)["smiles"].tolist()
        else:
            raise ValueError("Dataset not implemented")

        # ------------------------------------------------------------------
        # Extra / domain features
        # ------------------------------------------------------------------
        if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        # ------------------------------------------------------------------
        # Metrics & visualisation tools
        # ------------------------------------------------------------------
        if cfg.model.type == "discrete":
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {
            "dataset_infos": dataset_infos,
            "train_metrics": train_metrics,
            "sampling_metrics": sampling_metrics,
            "visualization_tools": visualization_tools,
            "extra_features": extra_features,
            "domain_features": domain_features,
        }
    else:
        raise ValueError("Unsupported dataset. Only molecular datasets are allowed in this clean version.")

    # ------------------------------------------------------------------
    # Handle resume / test‑only logic
    # ------------------------------------------------------------------
    if cfg.general.test_only:
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(os.path.dirname(cfg.general.test_only))
    elif cfg.general.resume:
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split("checkpoints")[0])

    # ------------------------------------------------------------------
    # Logging & checkpoint directories
    # ------------------------------------------------------------------
    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    # ------------------------------------------------------------------
    # Callbacks & Trainer setup
    # ------------------------------------------------------------------
    callbacks = []
    if cfg.train.save_model:
        ckpt_pattern = "gdpo-epoch{epoch}" if cfg.general.train_method in ["gdpo", "ddpo"] else "{epoch}"
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename=ckpt_pattern,
            every_n_epochs=1,
            save_top_k=-1,
            save_on_train_epoch_end=False,
        )
        callbacks.extend([
            ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename="last", every_n_epochs=1),
            checkpoint_callback,
        ])

    if cfg.train.ema_decay > 0:
        callbacks.append(utils.EMA(decay=cfg.train.ema_decay))

    trainer_kwargs = dict(
        accelerator="gpu" if torch.cuda.is_available() and cfg.general.gpus > 0 else "cpu",
        devices=cfg.general.gpus if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
        limit_train_batches=20 if cfg.general.name == "test" else None,
        limit_val_batches=20 if cfg.general.name == "test" else None,
        limit_test_batches=20 if cfg.general.name == "test" else None,
        val_check_interval=cfg.general.val_check_interval,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == "debug",
        strategy="ddp" if cfg.general.gpus > 1 else None,
        enable_progress_bar=cfg.general.train_method not in ["gdpo", "ddpo"],
        callbacks=callbacks,
        logger=[],
    )

    trainer = Trainer(**trainer_kwargs)

    # ------------------------------------------------------------------
    # Build model (fresh or resumed) and run train / test
    # ------------------------------------------------------------------
    if not cfg.general.test_only:
        if cfg.general.resume:
            model = DiscreteDenoisingDiffusion.load_from_checkpoint(
                cfg.general.resume,
                cfg=cfg,
                learning_rate=cfg.train.lr,
                amsgrad=cfg.train.amsgrad,
                weight_decay=cfg.train.weight_decay,
                **model_kwargs,
            )
            # Optional partial fine‑tuning ------------------------------------------------
            if cfg.general.partial and cfg.general.train_method in ["gdpo", "ddpo"] and "nodes" not in cfg.dataset:
                _freeze_transformer_layers(model, cfg)
            if cfg.dataset.name in ["zinc", "moses", "chembl"]:
                model.train_smiles = train_smiles
        else:
            model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
            # Optional partial fine‑tuning ------------------------------------------------
            if cfg.general.partial and cfg.general.train_method in ["gdpo", "ddpo"] and "nodes" not in cfg.dataset:
                _freeze_transformer_layers(model, cfg)
            # Optional initialisation from pretrained weights ---------------------------
            sd_dict = {
                "zinc": home_prefix + "pretrained/zincpretrained.pt",
                "moses": home_prefix + "pretrained/mosespretrained.pt",
                "chembl": home_prefix + "pretrained/chemblpretrained.pt",
                "guacamol": home_prefix + "pretrained/guacamolpretrained.pt",
            }
            if cfg.dataset.name in sd_dict and cfg.general.train_method in ["gdpo", "ddpo"]:
                _load_pretrained_transformer(model, sd_dict[cfg.dataset.name])
            if cfg.dataset.name in ["zinc", "moses", "chembl"]:
                model.train_smiles = train_smiles

        init(cfg)  # re‑seed workers after potential fork in PL
        trainer.fit(model, datamodule=datamodule)

        if cfg.general.name not in ["debug", "test"]:
            trainer.test(model, datamodule=datamodule)
    else:
        # ---------------- Test‑only branch ----------------
        cfg.general.test_method = "evalproperty"
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        if cfg.dataset.name in ["zinc", "moses", "chembl"]:
            model.train_smiles = train_smiles

        if str(cfg.general.test_only).endswith(".pt"):
            state_dict = torch.load(cfg.general.test_only, map_location="cpu")
            new_sd = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
            model.model.load_state_dict(new_sd)
            model.model.cuda()
            print("Loaded pretrained model from .pt checkpoint")
            trainer.test(model, datamodule=datamodule)
        else:
            model.ckpt = cfg.general.test_only
            trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)

        # Optional: evaluate all checkpoints in same folder
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            for file in os.listdir(directory):
                if file.endswith(".ckpt"):
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    global_step = torch.load(ckpt_path, map_location="cpu")["global_step"]
                    if global_step > 400:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    setup_wandb(cfg)
                    model.ckpt = ckpt_path
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

# ---------------------------------------------------------------------------
#  Utility helpers (kept outside main for clarity)
# ---------------------------------------------------------------------------

def _freeze_transformer_layers(model, cfg):
    """Freeze early transformer layers based on cfg.general.fix ratio."""
    for name, param in model.named_parameters():
        if "tf_layers" in name:
            layernum = int(name.split(".")[2])
        else:
            layernum = -1
        if "self_attn" not in name or layernum < int(cfg.general.fix * cfg.model.n_layers):
            param.requires_grad_(False)
        else:
            print("Unfrozen layer:", name)
    model.configure_optimizers()


def _load_pretrained_transformer(model, weight_path):
    """Load a subset of weights (prefixed by 'model.') into model.model."""
    sd = torch.load(weight_path, map_location="cpu")
    new_sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
    model.model.load_state_dict(new_sd, strict=False)
    model.model.cuda()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
