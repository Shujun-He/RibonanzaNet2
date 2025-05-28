import argparse
import time
import json
import sys
import os
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp
import grain.checkpoint
import grain.sharding
from tqdm import tqdm

import jax.distributed
import jax.sharding
import jax.tree_util

import ompi5_cluster


from load_and_split_data import load_and_split_rn2_ABCD
from csv_logger import CSVLogger
from yaml_config import load_config_from_yaml

import jax_dataset
import jax_network
import jax_ranger


def get_train_and_validation_data_loaders(config):
    hdf_file_paths, train_indices, val_indices = load_and_split_rn2_ABCD(config)
    assert config.nclass == 2 * len(hdf_file_paths)

    if config.use_data_percentage < 1:
        train_indices = train_indices[
            : int(len(train_indices) * config.use_data_percentage)
        ]
        val_indices = val_indices[: int(len(val_indices) * config.use_data_percentage)]
        if jax.process_index() == 0:
            print(f"Using {config.use_data_percentage} of data")
            print(f"Global train shape: {len(train_indices)}")
            print(f"Global val shape: {len(val_indices)}")

    if jax.process_index() == 0:
        print(f"Global train shape: {len(train_indices)}")
        print(f"Global val shape: {len(val_indices)}")

    # Calculate per-process batch sizes (data loader gets this)
    global_batch_size = config.batch_size
    global_test_batch_size = config.test_batch_size

    if global_batch_size % jax.process_count() != 0:
        raise ValueError(
            f"Global batch_size ({global_batch_size}) must be divisible by "
            f"process_count ({jax.process_count()})"
        )
    if global_test_batch_size % jax.process_count() != 0:
        raise ValueError(
            f"Global test_batch_size ({global_test_batch_size}) must be divisible by "
            f"process_count ({jax.process_count()})"
        )

    process_batch_size = global_batch_size // jax.process_count()
    process_test_batch_size = global_test_batch_size // jax.process_count()

    if jax.process_index() == 0:
        print(
            f"Global batch_size: {global_batch_size}, Per-process batch_size: {process_batch_size}"
        )
        print(
            f"Global test_batch_size: {global_test_batch_size}, Per-process test_batch_size: {process_test_batch_size}"
        )
        num_local_devices = jax.local_device_count()
        assert (
            process_batch_size % num_local_devices == 0
            and process_test_batch_size % num_local_devices == 0
        )

    shard_options_train = grain.sharding.ShardOptions(
        shard_index=jax.process_index(),
        shard_count=jax.process_count(),
        drop_remainder=True,
    )
    shard_options_val = grain.sharding.ShardOptions(
        shard_index=jax.process_index(),
        shard_count=jax.process_count(),
        drop_remainder=True,
    )

    # Pass process-level batch size to data loader
    # The data loader should yield batches of size `process_batch_size` for this process
    train_loader = jax_dataset.make_data_loader(
        hdf_file_paths=hdf_file_paths,
        indices=train_indices,
        random_flip=config.use_flip_aug,
        add_noise=config.use_noise_aug,
        batch_size=process_batch_size,
        shuffle=getattr(config, "shuffle_training_data", True),
        max_len=config.max_len,
        shard_options=shard_options_train,  # For Grain
    )

    val_loader = jax_dataset.make_data_loader(
        hdf_file_paths=hdf_file_paths,
        indices=val_indices,
        random_flip=False,
        add_noise=False,
        shuffle=False,
        epochs=1,
        max_len=config.max_len,
        batch_size=process_test_batch_size,
        shard_options=shard_options_val,  # For Grain
    )

    # batches_per_epoch is global steps per epoch using global batch size
    # This is used for LR schedule and epoch boundary checks.
    batches_per_epoch = len(train_indices) // global_batch_size

    # num_val_examples is global count, for progress bar total on root process
    # Effective val examples might be less if drop_remainder=True and not perfectly divisible
    effective_val_examples = (
        len(val_indices) // global_test_batch_size
    ) * global_test_batch_size

    return train_loader, val_loader, batches_per_epoch, effective_val_examples


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
    sn_weights = jnp.clip(SN, 0.5, 1.0)  # (batch, 1, num_exp*2)
    weighted_loss = elementwise_loss * sn_weights

    # Apply loss_masks
    # batch['loss_masks'] shape: (batch_size, seq_len, num_exp*2)
    masked_loss = jnp.where(loss_masks, weighted_loss, 0.0)

    # Mean over valid elements
    losses = jnp.sum(masked_loss, axis=[1, 2], keepdims=True) / jnp.sum(
        loss_masks, axis=[1, 2], keepdims=True
    )
    loss = jnp.mean(losses)

    return loss, losses


@nnx.jit
def train_step(train_state, batch, rngs: nnx.Rngs):
    grad_fn = nnx.value_and_grad(compute_training_loss, has_aux=True)
    (_, losses), grads = grad_fn(model=train_state.model.fast, batch=batch, rngs=rngs)
    train_state.metrics.update(values=losses)
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


def eval_step(model, metrics, batch, use_flip_aug: bool):
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
    loss = jnp.sum(masked_loss, axis=[1, 2], keepdims=True) / jnp.sum(
        loss_masks, axis=[1, 2], keepdims=True
    )
    metrics.update(values=loss)


class Trainer:
    def __init__(self, config):
        self.config = config

        self.mesh = jax.sharding.Mesh(devices=jax.devices(), axis_names=("data",))
        self.data_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec("data")
        )
        self.model_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec()
        )

        random_key = jax.random.key(getattr(config, "weight_initialization_seed", 0))
        (init_key, self.dropout_key) = jax.random.split(random_key)

        model = self._initialize_model(init_key=init_key)

        total_params = sum(
            p.size for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
        )
        # Data Loaders: Config batch sizes are global. get_... handles process-level.
        (
            train_loader,
            self.val_loader,
            self.batches_per_epoch,
            self.num_val_examples,
        ) = get_train_and_validation_data_loaders(config)  # config.batch_size is global

        if jax.process_index() == 0:
            self.logger = CSVLogger(
                ["epoch", "train_loss", "val_loss"], f"logs/fold{config.fold}.csv"
            )
        else:
            self.logger = None

        print(f"Total number of parameters in the model: {total_params}")

        self.checkpoint_manager = ocp.CheckpointManager(
            directory=getattr(
                config, "checkpoint_directory", os.path.abspath("checkpoints")
            ),
            options=ocp.CheckpointManagerOptions(
                create=True,
                save_interval_steps=config.log_interval,
                best_fn=lambda metrics: -metrics["validation_loss"],
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
        self.eval_step = jax.tree_util.Partial(
            eval_step, use_flip_aug=self.config.use_flip_aug
        )

    def _initialize_model(self, init_key):
        @nnx.jit
        def make_replicated_model():
            # Create model, unsharded initially
            model = jax_network.RibonanzaNet(
                self.config, rngs=nnx.Rngs(params=init_key)
            )
            state = nnx.state(model)
            pspecs = jax.tree.map(lambda _: self.model_sharding, state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(model, sharded_state)
            return model

        return make_replicated_model()

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
            if jax.process_index() == 0:
                print(f"Process 0: Restoring checkpoint from step {latest_step}")

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
            metrics={
                "training_loss": float(self.train_state.metrics.loss.compute()),
                "validation_loss": float(self.validation_metrics.loss.compute()),
            },
        )

    def _make_sharded_batch(self, local_batch, batch_size: int):
        return jax.tree.map(
            lambda local_data: jax.make_array_from_process_local_data(
                sharding=self.data_sharding,
                local_data=local_data,
                global_shape=(batch_size,) + local_data.shape[1:],
            ),
            local_batch,
        )

    def run_training(self):
        self.train_state.model.fast.train()
        tbar = None
        batch_size = self.config.batch_size

        def update_bar():
            nonlocal tbar
            step = int(self.train_state.step)
            epoch = (step - 1) // self.batches_per_epoch + 1
            step_within_epoch = step % self.batches_per_epoch
            if step_within_epoch == 0 and step > 0:
                step_within_epoch = self.batches_per_epoch
            if tbar is None:
                tbar = tqdm(
                    total=self.batches_per_epoch * batch_size,
                    disable=jax.process_index() != 0,
                    file=sys.stdout,
                )
                tbar.update(step_within_epoch * batch_size)
            else:
                tbar.update(batch_size)
            avg_loss = float(self.train_state.metrics.loss.compute())
            tbar.set_description(f"Epoch {epoch} Loss: {avg_loss:.05f}")

            print(f"Epoch {epoch} Loss: {avg_loss:.05f}")

        try:
            update_bar()
            for local_batch in self.training_data_loader:
                batch = self._make_sharded_batch(local_batch, batch_size=batch_size)
                train_step(
                    train_state=self.train_state,
                    batch=batch,
                    rngs=nnx.Rngs(
                        dropout=jax.random.fold_in(
                            self.dropout_key, self.train_state.step
                        )
                    ),
                )
                step = int(self.train_state.step)
                update_bar()
                if (step % self.batches_per_epoch) == 0:
                    if tbar is not None:
                        tbar.close()
                        tbar = None
                    self.run_validation()
                self.save_checkpoint()
                if (step % self.batches_per_epoch) == 0:
                    self.train_state.metrics.reset()
            self.save_checkpoint()
        finally:
            if tbar is not None:
                tbar.close()

    def run_validation(self):
        step = int(self.train_state.step)
        epoch = (step - 1) // self.batches_per_epoch + 1
        examples_per_batch = self.config.test_batch_size
        with tqdm(
            total=self.num_val_examples,
            disable=jax.process_index() != 0,
            file=sys.stdout,
        ) as tbar:
            self.validation_metrics.reset()
            model = self.train_state.model.slow
            model.eval()
            for batch in self.val_loader:
                self.eval_step(
                    model=model,
                    metrics=self.validation_metrics,
                    batch=self._make_sharded_batch(
                        batch, batch_size=examples_per_batch
                    ),
                )
                avg_loss = float(self.validation_metrics.loss.compute())
                tbar.update(examples_per_batch)
                tbar.set_description(f"Epoch: {epoch} Validation loss: {avg_loss:.05f}")


def main(args):
    if args.distributed:
        jax.distributed.initialize(
            cluster_detection_method=args.cluster_detection_method,
            coordinator_address=args.coordinator_address,
            process_id=args.process_id,
            num_processes=args.num_processes,
            coordinator_bind_address=args.coordinator_bind_address,
            initialization_timeout=args.initialization_timeout,
            slice_index=args.slice_index,
            local_device_ids=args.local_device_ids,
        )

    # Ensure JAX is aware of all devices across hosts early on
    if jax.process_index() == 0:
        print(f"JAX Global Device Count: {jax.device_count()}")
        print(f"JAX Local Device Count: {jax.local_device_count()}")
        print(f"JAX Process Count: {jax.process_count()}")
        print(f"JAX Process Index: {jax.process_index()}")

    start_time = time.time()

    config = load_config_from_yaml(args.config_path)
    if jax.process_index() == 0:
        config.print()

    # Create directories only on process 0 to avoid race conditions on shared filesystems
    if jax.process_index() == 0:
        os.makedirs("logs", exist_ok=True)
        # Checkpoint directory is created by Trainer if needed on process 0
        # Models and oofs dirs are not used by this training script directly for saving
        os.makedirs("models", exist_ok=True)
        os.makedirs("oofs", exist_ok=True)

    trainer = Trainer(config=config)

    if getattr(config, "load_torch_initial_state", False):
        import jax_torch_compare

        new_state = jax_torch_compare.load_torch_model_into_jax(
            "models/step_0/pytorch_model.bin", nnx.state(trainer.train_state.model.fast)
        )
        nnx.update(trainer.train_state.model.fast, new_state)
        nnx.update(trainer.train_state.model.slow, new_state)
    else:
        trainer.restore_checkpoint()

    trainer.run_training()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    with open("run_stats.json", "w") as file:
        json.dump({"Total_execution_time": elapsed_time}, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/pairwise.yaml")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run in multi-process distributed mode.",
    )
    parser.add_argument("--cluster-detection-method", type=str)
    parser.add_argument("--coordinator-address", type=str)
    parser.add_argument("--num-processes", type=int)
    parser.add_argument("--process-id", type=int)
    parser.add_argument("--coordinator-bind-address", type=int)
    parser.add_argument("--initialization-timeout", type=int)
    parser.add_argument("--slice-index", type=int)
    parser.add_argument("--local-device-ids", type=int, nargs="+")
    parsed_args = parser.parse_args()
    main(parsed_args)
