#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
import logging
import multiprocessing
import os
import sys
from functools import partial
import torch.distributed as dist

import torch
from nupic.torch.modules import rezero_weights, update_boost_strength
from ray.tune.resources import Resources
from ray.util.sgd.pytorch.pytorch_trainer import PyTorchTrainable, PyTorchTrainer
from ray.util.sgd.pytorch.utils import (
    SCHEDULER_STEP, SCHEDULER_STEP_BATCH, SCHEDULER_STEP_EPOCH
)
from torch.backends import cudnn

from nupic.research.frameworks.pytorch.imagenet.experiment_utils import (
    create_model,
    create_train_dataset,
    create_validation_dataset,
    create_optimizer,
    create_lr_scheduler
)
from nupic.research.frameworks.pytorch.model_utils import (
    train_model, evaluate_model,
    set_random_seed,
    count_nonzero_params
)


logger = logging.getLogger(__name__)


def model_creator(config):
    global logger
    logger.disabled = dist.is_initialized() and dist.get_rank() != 0

    model = create_model(
        model_class=config["model_class"],
        model_args=config.get("model_args", {}),
        init_batch_norm=config.get("init_batch_norm", False),
        checkpoint_file=config.get("checkpoint_file", None)
    )
    logger.debug(model)
    params_sparse, nonzero_params_sparse2 = count_nonzero_params(model)
    logger.debug("Params total/nnz %s / %s = %s ",
                 params_sparse, nonzero_params_sparse2,
                 float(nonzero_params_sparse2) / params_sparse)
    return model


def data_creator(config):
    data_dir = config["data"]
    num_classes = config.get("num_classes", 1000)
    train_dataset = create_train_dataset(
        data_dir=data_dir,
        train_dir=config.get("train_dir", "train"),
        num_classes=num_classes,
        use_auto_augment=config.get("use_auto_augment", False))
    val_dataset = create_validation_dataset(
        data_dir=data_dir,
        val_dir=config.get("val_dir", "val"),
        num_classes=num_classes)

    return train_dataset, val_dataset


def optimizer_creator(model, config):
    optimizer_class = config.get("optimizer_class", torch.optim.SGD)
    optimizer_args = config.get("optimizer_args", {})
    batch_norm_weight_decay = config.get("batch_norm_weight_decay", True)
    return create_optimizer(
        model=model,
        optimizer_class=optimizer_class,
        optimizer_args=optimizer_args,
        batch_norm_weight_decay=batch_norm_weight_decay,
    )


def scheduler_creator(optimizer, config):
    lr_scheduler_class = config.get("lr_scheduler_class", None)
    if lr_scheduler_class is not None:
        lr_scheduler_args = config.get("lr_scheduler_args", {})
        steps_per_epoch = config.get("steps_per_epoch", None)
        return create_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_args=lr_scheduler_args,
            steps_per_epoch=steps_per_epoch)
    return None


def pre_epoch(model):
    model.apply(update_boost_strength)


def post_batch(model, loss, batch_idx, num_images, time_string, scheduler=None):
    # Update 1cycle learning rate after every batch
    if scheduler:
        scheduler.step()


def post_epoch(model, post_epoch_hooks=None, scheduler=None):
    global logger
    logger.disabled = dist.is_initialized() and dist.get_rank() != 0

    count_nnz = logger.isEnabledFor(logging.DEBUG)
    if count_nnz:
        params_sparse, nonzero_params_sparse1 = count_nonzero_params(model)

    model.apply(rezero_weights)
    if post_epoch_hooks:
        for hook in post_epoch_hooks:
            model.apply(hook)

    if count_nnz:
        params_sparse, nonzero_params_sparse2 = count_nonzero_params(model)
        logger.debug("Params total/nnz before/nnz after %s %s / %s = %s",
                     params_sparse, nonzero_params_sparse1,
                     nonzero_params_sparse2,
                     float(nonzero_params_sparse2) / params_sparse)

    # Update learning rate
    if scheduler:
        scheduler.step()


def train_function(config, model, loader, criterion, optimizers, scheduler):
    post_batch_scheduler = None
    post_epoch_scheduler = None

    if scheduler:
        scheduler_step = config.get(SCHEDULER_STEP, SCHEDULER_STEP_EPOCH)
        if scheduler_step == SCHEDULER_STEP_BATCH:
            post_batch_scheduler = scheduler
        elif scheduler_step == SCHEDULER_STEP_EPOCH:
            post_epoch_scheduler = scheduler

    pre_epoch(model)

    train_model(
        model=model,
        loader=loader,
        optimizer=optimizers,
        criterion=criterion,
        batches_in_epoch=config.get("batches_in_epoch", sys.maxsize),
        post_batch_callback=partial(post_batch, scheduler=post_batch_scheduler))

    post_epoch(
        model=model,
        post_epoch_hooks=config.get("post_epoch_hooks", []),
        scheduler=post_epoch_scheduler
    )

    return {}

def validation_function(config, model, loader, criterion, scheduler=None):
    return evaluate_model(
        model=model,
        loader=loader,
        batches_in_epoch=config.get("batches_in_epoch", sys.maxsize),
        criterion=criterion)


def configure_logger(config):
    # Configure logging related stuff
    global logger
    log_format = config.get("log_format", logging.BASIC_FORMAT)
    log_level = getattr(logging, config.get("log_level", "INFO").upper())
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger(config.get("name", __name__))
    logger.setLevel(log_level)
    logger.addHandler(console)


def update_env_vars(config):
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def initialization_hook(runner):
    config = runner.config
    update_env_vars(config)
    configure_logger(config)

    # Improves performance when using fixed size images (224) and CNN
    cudnn.benchmark = True

    # CUDA runtime does not support the fork start method.
    # See https://pytorch.org/docs/stable/notes/multiprocessing.html
    if torch.cuda.is_available():
        multiprocessing.set_start_method("spawn")

    # Configure seed
    seed = config.get("seed", 42)
    set_random_seed(seed, False)


class ImagenetRaySGDTrainable(PyTorchTrainable):

    @classmethod
    def default_resource_request(cls, config):
        num_gpus = config.get("num_gpus", 0)
        num_cpus = config.get("num_cpus", 1)
        use_gpu = num_gpus > 0
        if use_gpu:
            num_replicas = num_gpus
        else:
            num_replicas = num_cpus
        return Resources(
            cpu=0,
            gpu=0,
            extra_cpu=num_replicas,
            extra_gpu=int(use_gpu) * num_replicas)

    def __init__(self, config=None, logger_creator=None):
        super().__init__(config, logger_creator)
        self.epochs = 1
        self.epochs_to_validate = []

    def _setup(self, config):
        num_gpus = config.get("num_gpus", 0)
        num_cpus = config.get("num_cpus", 1)
        use_gpu = num_gpus > 0
        if use_gpu:
            num_replicas = num_gpus
        else:
            num_replicas = num_cpus

        self.epochs = config.get("epochs", 1)
        self.epochs_to_validate = config.get("epochs_to_validate",
                                             range(self.epochs - 3, self.epochs + 1))

        self._trainer = PyTorchTrainer(
            model_creator=model_creator,
            data_creator=data_creator,
            optimizer_creator=optimizer_creator,
            loss_creator=config.get("loss_function", torch.nn.CrossEntropyLoss),
            scheduler_creator=scheduler_creator,
            train_function=train_function,
            validation_function=validation_function,
            initialization_hook=initialization_hook,
            config=config,
            dataloader_config=dict(
                num_workers=config.get("workers", 0),
                pin_memory=torch.cuda.is_available()
            ),
            num_replicas=num_replicas,
            use_gpu=use_gpu,
            batch_size=config.get("batch_size", 1),
            backend=config.get("backend", "auto"),
            use_fp16=config.get("mixed_precision", False),
            apex_args=config.get("mixed_precision_args", None),
            scheduler_step_freq=config.get("scheduler_step_freq", SCHEDULER_STEP_BATCH))

    def _train(self):
        train_stats = self._trainer.train()
        if self.iteration in self.epochs_to_validate:
            validation_stats = self._trainer.validate()
            train_stats.update(validation_stats)

        return train_stats
