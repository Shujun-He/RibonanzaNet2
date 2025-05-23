import argparse
import time
import json
import os
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp
import grain
from tqdm import tqdm

from load_and_split_data import load_and_split_rn2_ABCD
from csv_logger import CSVLogger
from yaml_config import load_config_from_yaml

import jax_dataset
import jax_network
import jax_ranger


def get_train_and_validation_data_loaders(config):
    hdf_file_paths, train_indices, val_indices = load_and_split_rn2_ABCD(config)
    config.num_output_channels = 2 * len(
        hdf_file_paths
    )  # Set based on actual loaded data

    if config.use_data_percentage < 1:
        train_indices = train_indices[
            : int(len(train_indices) * config.use_data_percentage)
        ]
        val_indices = val_indices[: int(len(val_indices) * config.use_data_percentage)]
        print(f"Using {config.use_data_percentage} of data")
        print(f"train shape: {len(train_indices)}")
        print(f"val shape: {len(val_indices)}")

    print(f"Train shape: {len(train_indices)}")
    print(f"Val shape: {len(val_indices)}")

    train_loader = jax_dataset.make_data_loader(
        hdf_file_paths=hdf_file_paths,
        indices=train_indices,
        random_flip=config.use_flip_aug,
        add_noise=config.use_noise_aug,
        batch_size=config.batch_size,
        shuffle=getattr(config, "shuffle_training_data", True),
        max_len=config.max_len,
    )

    val_loader = jax_dataset.make_data_loader(
        hdf_file_paths=hdf_file_paths,
        indices=val_indices,
        random_flip=False,
        add_noise=False,
        shuffle=False,
        epochs=1,
        max_len=config.max_len,
        batch_size=config.test_batch_size,
    )

    batches_per_epoch = len(train_indices) // config.batch_size

    return train_loader, val_loader, batches_per_epoch


def get_optimizer(model, config, batches_per_epoch: int):
    lr_schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=batches_per_epoch // config.gradient_accumulation_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.clip_grad_norm),
        jax_ranger.ranger(weight_decay=config.weight_decay, lr=lr_schedule),
    )

    if config.gradient_accumulation_steps > 1:
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=config.gradient_accumulation_steps
        )

    params = optax.LookaheadParams(fast=model, slow=nnx.clone(model))

    return optimizer, params


def compute_training_loss(model, batch, rngs: nnx.Rngs):
    src = batch["sequence"]
    masks = batch["masks"]
    labels = batch["labels"]
    SN = batch["SN"]

    loss_masks = batch["loss_masks"]
    SN = SN[:, None, :]
    loss_masks = loss_masks * (SN >= 0.5)

    preds = model(src, masks, deterministic=False, rngs=rngs)
    elementwise_loss = jnp.abs(preds - labels)

    # weight with SN, downweight low quality data, and high
    # quality data has up to 1 weight
    #
    # batch['SN'] shape: (batch_size, num_experiments*2)
    # labels shape: (batch_size, seq_len, num_experiments*2)
    sn_weights = jnp.clip(batch["SN"][:, None, :], 0.5, 1.0)  # (batch, 1, num_exp*2)
    weighted_loss = elementwise_loss * sn_weights

    # Apply loss_masks
    # batch['loss_masks'] shape: (batch_size, seq_len, num_exp*2)
    masked_loss = jnp.where(loss_masks, weighted_loss, 0.0)

    # Mean over valid elements
    loss = jnp.sum(masked_loss) / jnp.sum(loss_masks)
    return loss


@nnx.jit
def train_step(train_state, batch, rngs: nnx.Rngs):
    grad_fn = nnx.value_and_grad(compute_training_loss)
    loss, grads = grad_fn(model=train_state.model.fast, batch=batch, rngs=rngs)
    train_state.metrics.update(values=loss)
    train_state.update(grads)


def flip_sequences(source, length):
    max_len = source.shape[1]
    return jnp.stack(
        [
            jnp.roll(
                jnp.flip(src_part, axis=0),
                shift=src_len - max_len,
                axis=0,
            )
            for src_part, src_len in zip(source, length)
        ],
        axis=0,
    )


def get_eval_step(use_flip_aug: bool):
    @nnx.jit
    def eval_step(model, metrics, batch):
        src = batch["sequence"]
        masks = batch["masks"]
        labels = batch["labels"]
        loss_masks = batch["loss_masks"]
        length = batch["length"]

        preds = model(src, masks, deterministic=True)

        if use_flip_aug:
            src_flipped = flip_sequences(src, length)
            preds_flipped = model(src_flipped, masks, deterministic=True)
            preds = (preds + flip_sequences(preds_flipped, length)) / 2

        # L1 loss, element-wise for eval (no SN weighting here as per original val_criterion)
        elementwise_loss = jnp.abs(preds - labels)
        masked_loss = jnp.where(loss_masks, elementwise_loss, 0.0)
        loss = jnp.sum(masked_loss) / jnp.sum(loss_masks)

        return loss

    return eval_step


class Trainer:
    def __init__(self, config):
        self.config = config
        train_loader, self.val_loader, self.batches_per_epoch = (
            get_train_and_validation_data_loaders(config)
        )
        self.logger = CSVLogger(
            ["epoch", "train_loss", "val_loss"], f"logs/fold{config.fold}.csv"
        )

        random_key = jax.random.key(getattr(config, "weight_initialization_seed", 0))
        (init_key, self.dropout_key) = jax.random.split(random_key)

        model = jax_network.RibonanzaNet(config, rngs=nnx.Rngs(params=init_key))
        self.model = model

        total_params = sum(
            p.size for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
        )
        print(f"Total number of parameters in the model: {total_params}")

        self.checkpoint_manager = ocp.CheckpointManager(
            directory=getattr(
                config, "checkpoint_directory", os.path.abspath("checkpoints")
            ),
            options=ocp.CheckpointManagerOptions(
                create=True,
                save_interval_steps=config.log_interval,
            ),
        )

        self.optimizer, params = get_optimizer(
            model=model,
            config=config,
            batches_per_epoch=self.batches_per_epoch,
        )

        self.train_state = nnx.Optimizer(tx=self.optimizer, model=params)
        self.train_state.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average(),
        )
        self.training_data_loader = iter(train_loader)
        self.validation_metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average(),
        )
        self.eval_step = get_eval_step(self.config.use_flip_aug)

    def _get_checkpoint_state(self):
        return {
            "metrics": self.train_state.metrics,
            "step": self.train_state.step,
            "params": self.train_state.model.slow,
            "fast_params": self.train_state.model.fast,
            "opt_state": self.train_state.opt_state,
        }

    def restore_checkpoint(self):
        latest_step = self.checkpoint_manager.latest_step()
        if latest_step is not None:
            print(f"restoring latest_step={latest_step}")
            checkpoint_state = self._get_checkpoint_state()

            restored = self.checkpoint_manager.restore(
                step=latest_step,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(nnx.state(checkpoint_state)),
                    training_data_input=grain.checkpoint.CheckpointSave(
                        self.training_data_loader
                    ),
                ),
            )
            self.training_data_loader = restored.training_data_input
            nnx.update(checkpoint_state, restored.state)

    def save_checkpoint(self):
        checkpoint_state = self._get_checkpoint_state()
        self.checkpoint_manager.save(
            step=int(self.train_state.step),
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(nnx.state(checkpoint_state)),
                training_data_input=grain.checkpoint.CheckpointRestore(
                    self.training_data_loader
                ),
            ),
        )

    def run_training(self):
        self.train_state.model.fast.train()
        for batch in self.training_data_loader:
            print(batch["SN"])
            train_step(
                train_state=self.train_state,
                batch=batch,
                # FIXME include jax process number also
                rngs=nnx.Rngs(
                    dropout=jax.random.fold_in(self.dropout_key, self.train_state.step)
                ),
            )
            print(
                f"performed step={int(self.train_state.step)} loss={float(self.train_state.metrics.loss.compute())}"
            )
            self.save_checkpoint()
            if (self.train_state.step % self.batches_per_epoch) == 0:
                self.run_validation()
        self.save_checkpoint()

    def run_validation(self):
        tbar = tqdm(self.val_loader)
        print("doing val")
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average(),
        )
        model = self.train_state.model.slow
        model.eval()
        for batch in tbar:
            loss = self.eval_step(
                model=model,
                metrics=metrics,
                batch=batch,
            )
            metrics.update(values=loss)


# --- Main Script ---
def main(args):
    start_time = time.time()

    config = load_config_from_yaml(args.config_path)
    config.print()

    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("oofs", exist_ok=True)

    trainer = Trainer(config=config)
    # trainer.restore_checkpoint()
    import jax_torch_compare

    new_state = jax_torch_compare.load_torch_model_into_jax(
        "models/step_0/pytorch_model.bin", nnx.state(trainer.train_state.model.fast)
    )
    nnx.update(trainer.train_state.model.fast, new_state)
    nnx.update(trainer.train_state.model.slow, new_state)
    trainer.run_training()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    with open("run_stats.json", "w") as file:
        json.dump({"Total_execution_time": elapsed_time}, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/pairwise.yaml")
    parsed_args = parser.parse_args()
    main(parsed_args)
