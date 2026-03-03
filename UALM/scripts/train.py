# Copyright (c) 2026 NVIDIA CORPORATION.
#   Licensed under the MIT license.

#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training script for UALM with distributed training support."""

import argparse
import logging
import os
from pathlib import Path
import sys
import time
time.sleep(10)

import deepspeed
import torch
import wandb
import yaml

from dataloader.iterator import DataIteratorFactory
from models import _all_job_types
from trainer.deepspeed_trainer import DeepSpeedTrainer


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser for training script."""
    parser = argparse.ArgumentParser(
        description="UALM Distributed Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Distributed training
    dist_group = parser.add_argument_group("Distributed Training")
    dist_group.add_argument(
        "--local_rank",
        type=int,
        # default=None,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local rank for distributed training (set by launcher)",
    )

    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--train-config",
        type=Path,
        required=True,
        help="Path to training configuration file",
    )
    train_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/train"),
        help="Directory to save checkpoints and logs",
    )
    train_group.add_argument(
        "--resume_path",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--train-unregistered-specifier",
        type=str,
        default="",
        required=False,
        help="Unregistered train data specifier. "
        "Format: 'task:name:data_json[:factor]' "
        "(e.g., 'asr:librispeech:train.json:2.0')",
    )
    data_group.add_argument(
        "--train-registered-specifier",
        type=str,
        default="",
        required=False,
        help="Registered train data specifier. "
        "Format: 'task:name[:factor]' "
        "(e.g., 'tts:ljspeech:1.5')",
    )
    data_group.add_argument(
        "--valid-unregistered-specifier",
        type=str,
        default="",
        required=False,
        help="Unregistered validation data specifier. "
        "Format: 'task:name:data_json[:factor]' "
        "(e.g., 'asr:librispeech:valid.json')",
    )
    data_group.add_argument(
        "--valid-registered-specifier",
        type=str,
        default="",
        required=False,
        help="Registered validation data specifier. "
        "Format: 'task:name[:factor]' "
        "(e.g., 'tts:ljspeech:1.0')",
    )
    data_group.add_argument(
        "--data-config-path",
        type=str,
        default="",
        required=False,
        help="configuration path of datasets; no need for registered or unregistered specifiers",
    )
    data_group.add_argument(
        "--stats-dir",
        type=Path,
        required=True,
        help="The folder of length statistics",
    )

    # Logging configuration
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    # Wandb configuration
    wandb_group = parser.add_argument_group("Weights & Biases Configuration")
    wandb_group.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Wandb logging mode (online=sync to cloud, offline=local only)",
    )
    wandb_group.add_argument(
        "--wandb-project",
        type=str,
        default="ualm",
        help="Project name for wandb",
    )
    wandb_group.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Run name for wandb (defaults to output dir name)",
    )
    wandb_group.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=None,
        help="Tags for organizing runs (e.g., baseline, v2, ablation)",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # (1) Setup distributed training first to get rank info
    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed()

    assert torch.distributed.is_initialized()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # (2) Setup logging with rank-aware configuration
    log_format = (
        f"[Rank {rank}/{world_size}] "
        "%(asctime)s (%(module)s:%(lineno)d) "
        "%(levelname)s: %(message)s"
    )

    if rank == 0:
        log_level = args.log_level
    else:
        log_level = "CRITICAL"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    logger.info("Distributed training initialized")
    logger.info(f"World size: {world_size}")
    logger.info(f"Output directory: {args.output_dir}")

    # (3) Initialize job template
    with open(args.train_config, "r") as f:
        train_config = yaml.safe_load(f)
    logger.info(f"Loaded training config from: {args.train_config}")

    job_template_class = _all_job_types[train_config["job_type"]]
    job_template = job_template_class(train_config, is_train=True)

    # (4) build data iterator factory
    loading_config = train_config["data_loading"]
    preprocessor = job_template.build_preprocessor()

    loader_state_dir = args.output_dir / "loader_state"
    loader_state_dir.mkdir(parents=True, exist_ok=True)

    # Setup Registry for CombinedDataset

    DATA_CONFIG_PATH = Path(args.data_config_path)
    print(f"Data config path: {DATA_CONFIG_PATH}")
    
    manifests = list(DATA_CONFIG_PATH.glob("*_manifest.json"))
    stats_files = list((DATA_CONFIG_PATH / "stats").glob("*.jsonl"))
    metadata_lmdb = DATA_CONFIG_PATH / "_metadata.lmdb"

    registry_path = DATA_CONFIG_PATH / "dataset_registry.yaml"

    # ---- rank 0: load/create/update registry ----
    if rank == 0:
        registry_content = {}
        if registry_path.exists():
            with open(registry_path, "r") as f:
                print(f"Loading registry from {registry_path}")
                loaded = yaml.safe_load(f) or {}
                print(f"Loaded registry: {loaded}")
            
            if isinstance(loaded, dict):
                registry_content = loaded
            else:
                registry_content = {}
        
        if not registry_content:
            for m in manifests:
                ds_id = m.stem.replace("_manifest", "")
                registry_content[ds_id] = {"path": str(m.resolve())}

            registry_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
            with open(tmp_path, "w") as f:
                yaml.safe_dump(registry_content, f, sort_keys=True)
            os.replace(tmp_path, registry_path)
            print(f"Created/updated registry at {registry_path}")

    # everyone waits until rank 0 is done
    torch.distributed.barrier()

    # Now all ranks can safely use it
    os.environ["ESPNET_DATASET_REGISTRY"] = str(registry_path)
    

    # Construct Specifier String for ALL datasets
    specifiers = []

    for s_file in stats_files:
        # Filename: stats_{task}_{id}.jsonl
        fname = s_file.stem # stats_...
        
        # Find which dataset ID is at the end of this string
        matched_ds_id = None
        for m in manifests:
            ds_id = m.stem.replace("_manifest", "")
            if fname.endswith(ds_id):
                matched_ds_id = ds_id
                break
                
        if matched_ds_id:
            # Extract task: stats_{task}_{id} -> remove prefix and suffix
            task_part = fname[6 : -len(matched_ds_id)-1] # stats_..._ID
            specifiers.append(f"{task_part}:{matched_ds_id}:1.0")

    full_specifier = " ".join(specifiers)
    print(f"\nInitializing Factory with specifier:\n  {full_specifier}")

    train_iterator_factory = DataIteratorFactory(
        registered_specifier=full_specifier,
        stats_dir=args.stats_dir,
        loader_state=loader_state_dir / f"train_{rank}_{world_size}.json",
        save_loader_state=True,
        collate_fn=preprocessor.collate_fn,
        batchfy_method=loading_config["batchfy_method"],
        batch_size=loading_config["batch_size"],
        num_workers=loading_config["num_workers"],  # if num_workers > 0: RuntimeError: Failed to load dataset: cannot pickle 'Environment' object
        rank=rank,
        world_size=world_size,
        shuffle=True,
        seed=loading_config["seed"],
    )

    valid_iterator_factories = dict()
    valid_iterator_args = dict(
        stats_dir=args.stats_dir,
        collate_fn=preprocessor.collate_fn,
        batchfy_method=loading_config["batchfy_method"],
        batch_size=loading_config["batch_size"],
        num_workers=loading_config["num_workers"],
        rank=rank,
        world_size=world_size,
        shuffle=False,
    )

    for spec in args.valid_unregistered_specifier.split():
        factory = DataIteratorFactory(
            unregistered_specifier=spec, **valid_iterator_args
        )
        valid_iterator_factories[spec] = factory
    for spec in args.valid_registered_specifier.split():
        factory = DataIteratorFactory(registered_specifier=spec, **valid_iterator_args)
        valid_iterator_factories[spec] = factory

    # (5) build model
    model = job_template.build_model()

    # (6) Initialize wandb: on rank 0 GPU
    wandb_name = args.wandb_name or f"run_{args.output_dir.name}"
    if rank == 0:
        wandb_argument_record = {
            "train_args": vars(args),
            "train_config": train_config,
        }
        wandb.init(
            mode=args.wandb_mode,
            project=args.wandb_project,
            name=wandb_name,
            config=wandb_argument_record,
            tags=args.wandb_tags,
            dir=str(args.output_dir),
            resume="auto",
        )
    else:
        wandb.init(mode="disabled")
    logger.info(
        f"wandb initialization: mode={args.wandb_mode}, "
        f"project={args.wandb_project}, name={wandb_name}"
    )

    # (7) Initialize DeepSpeed trainer and train
    trainer = DeepSpeedTrainer(
        train_data_factory=train_iterator_factory,
        valid_data_factories=valid_iterator_factories,
        model=model,
        resume_path=args.resume_path,
        output_dir=args.output_dir,
        trainer_args=train_config["trainer"],
    )
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
