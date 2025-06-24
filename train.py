# modified from stable-audio-tools under the MIT license

import argparse
import json
import os, sys
import torch
import pytorch_lightning as pl
import random

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict

from stable_audio_tools.utils.addict import Dict as AttrDict

from datetime import timedelta
from typing import Any, Dict, Optional, Union
from typing_extensions import override
from pprint import pformat

class EarlyStoppingCallback(pl.callbacks.EarlyStopping):
    """
    simple EarlyStopping wrappter that _run_early_stopping_check for every training steps
    useful to terminate job right after we face nan/inf loss, etc.
    NOTE: you probably want to use this using large patience (e.g. 10M) and check_on_train_epoch_end=False since this calls the check every training step (NOT epoch-level)!
    """
    @override
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: "pl.utilities.types.STEP_OUTPUT", batch: Any, batch_idx: int
    ) -> None:
        self._run_early_stopping_check(trainer)

    @override
    def _run_early_stopping_check(
        self, trainer: "pl.Trainer"
    ) -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(logs):  # disable early_stopping with fast_dev_run
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
            self._terminate_with_error_code(trainer)
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _terminate_with_error_code(
        self, trainer: "pl.Trainer"
    ) -> None:
        """Terminates the training process with a non-zero return code to signal an error."""
        print(f"[ERROR] Training force-stopped due to early stopping criteria being met. Usually it means nan/inf training loss is detected.")
        sys.exit(1)  # Terminate the process with a non-zero return code to signal an error
        

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')
        
        
class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main(args):
    seed = args.seed
    
    # huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    # To disable this warning, you can either:
    #         - Avoid using `tokenizers` before the fork if possible
    #         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    random.seed(seed)
    torch.manual_seed(seed)

    # Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)
        
    # convert it to AttrDict (dot-accessible dictionary)
    model_config = AttrDict(model_config)
    # override attr_dict with hparams
    if args.params != []:
        model_config.update_params(args.params)
   
    ###############################################################
    # hacky handling args hyperparams overwritten from args.params
    ###############################################################
    # Check for conflicts between args and model_config for the three attributes
    conflicting_keys = ['pretrained_ckpt_path', 'remove_pretransform_weight_norm', 'pretransform_ckpt_path', 'wrapper_ckpt_path']
    for key in conflicting_keys:
        if getattr(args, key) and getattr(model_config, key):
            raise AssertionError(f"{key} should only be declared from either args.params or argparse directly, not both.")

    # Assign attributes from args if provided, otherwise from model_config
    args.pretrained_ckpt_path = args.pretrained_ckpt_path or model_config.get('pretrained_ckpt_path', '')
    args.remove_pretransform_weight_norm = args.remove_pretransform_weight_norm or model_config.get('remove_pretransform_weight_norm', '')
    args.pretransform_ckpt_path = args.pretransform_ckpt_path or model_config.get('pretransform_ckpt_path', '')
    args.wrapper_ckpt_path = args.wrapper_ckpt_path or model_config.get('wrapper_ckpt_path', '')
    
    # Also override other user-defined args hyperparams if specified in model_config updated from args.params
    # Define the valid precision values
    valid_precisions = ['32-true', 'bf16-mixed', '16-mixed']
    # List of possible model config keys and their corresponding args attributes for overriding
    config_keys = ['save_dir', 'precision', 'batch_size', 'num_gpus', 'num_nodes', 'gradient_clip_val', 'checkpoint_every']
    for key in config_keys:
        if key in model_config:
            setattr(args, key, str(model_config[key]) if key in ['precision', 'save_dir'] else model_config[key])
    # Validate the precision value
    assert args.precision in valid_precisions, f"Select precision only from {valid_precisions}! Got {args.precision}."
    ###############################################################

    if model_config.model_type != "autoencoder" and not args.pretransform_ckpt_path:
        raise ValueError(f"args.pretransform_ckpt_path not specified for models that require it: {model_config.model_type}")
    
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)    
        
    train_dl, list_valid_dl = create_dataloader_from_config(
        dataset_config,
        args.dataset_config,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    model = create_model_from_config(model_config)
    
    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        print(f"loading pretransform_ckpt_path: {args.pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
    
    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)
    
    # load pretrained_ckpt_path if specified. moved this after pretrainsform_ckpt_path to load possibly WN-removed pretransform from pretrained_ckpt_path
    if args.pretrained_ckpt_path:
        if args.pretransform_ckpt_path:
            print(f"[WARNING(train.py)] args.pretrained_ckpt_path {args.pretrained_ckpt_path} will override pretransform weight of args.pretransform_ckpt_path {args.pretransform_ckpt_path}")
        print(f"loading pretrained_ckpt_path: {args.pretrained_ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    training_wrapper = create_training_wrapper_from_config(model_config, model)
    
    # load wrapped ckpt for 2nd-stage training exps: swapping out quantizers from pretrained model & discrimiantor...
    # This still resets the optimizers
    if args.wrapper_ckpt_path:
        print(f"loading wrapper_ckpt_path: {args.wrapper_ckpt_path}")
        copy_state_dict(training_wrapper, load_ckpt_state_dict(args.wrapper_ckpt_path))
    
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=args.save_dir, name=args.name, version='', sub_dir="tb_logs")

    exc_callback = ExceptionCallback()
    
    checkpoint_dir = os.path.join(args.save_dir, args.name)
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every,
        dirpath=checkpoint_dir,
        save_top_k=10,
        monitor="global_step",
        mode="max"
        
    )
    train_time_interval_minutes = 60 # save checkpoint every hour
    timed_ckpt_callback = pl.callbacks.ModelCheckpoint(
        train_time_interval=timedelta(minutes=train_time_interval_minutes),
        dirpath=checkpoint_dir,
        save_last='link'
    )
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)
    
    # safeguard early stopping callback that cancels training if nan/inf is detected. prevents corrupting saved checkpoints
    early_stopping_callback = EarlyStoppingCallback(
        monitor="train/loss", # NOTE: monitors final scalar training loss for every training batch
        patience=10000000, # arbitrary high number (10M), disables patience-based early stopping
        check_finite=True, # all we're interested is checking inf/nan loss with this
        strict=False, # to support autoencoder which does not return "train/loss" for discriminator update step (returns for generator update steps)
        check_on_train_epoch_end=False # disable per-epoch checking
    )

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2, 
                                        contiguous_gradients=True, 
                                        overlap_comm=True, 
                                        reduce_scatter=True, 
                                        reduce_bucket_size=5e8, 
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True
                                        )
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto" 

    # Combine args and model_config dicts so far used. This defines all params except dataset_config
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    print(pformat(args_dict))
    
    # Save args_dict and dataset_config to separate JSON files for future reference
    if not os.path.exists(os.path.join(args.save_dir, args.name)):
        os.makedirs(os.path.join(args.save_dir, args.name), exist_ok=True)
    with open(os.path.join(args.save_dir, args.name, "config.json"), 'w') as config_file:
        json.dump(args_dict, config_file, indent=4)
    with open(os.path.join(args.save_dir, args.name, "dataset_config.json"), 'w') as dataset_config_file:
        json.dump(dataset_config, dataset_config_file, indent=4)        

    # define trainer
    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, timed_ckpt_callback, demo_callback, exc_callback, save_model_config_callback, early_stopping_callback],
        logger=tensorboard_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=None,
        val_check_interval=model_config.training.demo.demo_every, # use demo_every for validation loss tracking.
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val, # unsupported for manual optimization
        reload_dataloaders_every_n_epochs = 0,
        enable_progress_bar=args.enable_progress_bar
    )
    
    # start training!
    trainer.fit(
        model=training_wrapper,
        train_dataloaders=train_dl,
        val_dataloaders=list_valid_dl if list_valid_dl != [] else None,
        ckpt_path=args.ckpt_path if args.ckpt_path else None
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")

    # General settings
    parser.add_argument('--name', type=str, default='stable_audio_tools', help='Name of the run')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs to use for training')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes to use for training')
    parser.add_argument('--strategy', type=str, default='', help='Multi-GPU strategy for PyTorch Lightning')
    parser.add_argument('--precision', type=str, default='32-true', help='Precision to use for training')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of CPU workers for the DataLoader')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--accum_batches', type=int, default=1, help='Batches for gradient accumulation')
    parser.add_argument('--checkpoint_every', type=int, default=50000, help='Number of steps between checkpoints')
    parser.add_argument('--enable_progress_bar', type=bool, default=False, help='whether to turn on tqdm progbar')
    
    # Checkpoint paths
    parser.add_argument('--ckpt_path', type=str, default='', help='Trainer checkpoint file to restart training from')
    parser.add_argument('--pretrained_ckpt_path', type=str, default='', help='Model checkpoint file to start a new training run from')
    parser.add_argument('--wrapper_ckpt_path', type=str, default='', help='Wrapper checkpoint file to restart training from (resets optimizer). Must be the one with its training wrapper which did not apply unwrap_model.py. Useful for 2nd-stage training (e.g. plugging in quantizer into VAE, etc.)')
    parser.add_argument('--pretransform_ckpt_path', type=str, default='', help='Checkpoint path for the pretransform model if needed')

    # Configuration files
    parser.add_argument('--model_config', type=str, default='', help='Configuration model specifying model hyperparameters')
    parser.add_argument('--dataset_config', type=str, default='', help='Configuration for datasets')

    # Storage directories
    parser.add_argument('--save_dir', type=str, default='exp', help='Directory to save the checkpoints in')

    # Training settings
    parser.add_argument('--gradient_clip_val', type=float, default=0.0, help='Gradient clip value passed into PyTorch Lightning Trainer')
    parser.add_argument('--remove_pretransform_weight_norm', type=str, default='', help='Remove the weight norm from the pretransform model')
    
    # Extra parameters
    parser.add_argument('--params', nargs='+', default=[], help='Additional params for overriding args and model_config.json')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 99999)
        print(f"####################################################################")
        print(f"[WARNING(train.py)]: seed not specified. randomizing seed for this training run: {args.seed}")
        print(f"####################################################################")
    main(args)