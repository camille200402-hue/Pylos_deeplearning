# Training the Pylos networks

Java self-play produces `.pdat` files; Rust trains the networks and saves their
parameters. Start with **value** (predicting game outcomes), then **policy**
(learning action probabilities from MCTS visits).

## 1. Install Rust

Install stable Rust by following the instructions on
[rustup](https://rust-lang.org/tools/install/).

## 2. Collect data

With the Java project configured for **JDK 21**:

1. Run [PylosMLCollect.main](../java/be/kuleuven/pylos/PylosMLCollect.java) in your
   IDE with the **repository root** as its working directory.
2. Threads are set automatically. Increase `GAMES_PER_THREAD` to collect more
   data in one run.
3. Wait for completion, or press Ctrl+C and let active games finish.

Completed `.pdat` files appear in
`pylos-ml/src/main/training/data/run_<timestamp>/`. Files ending in `.partial`
are still unfinished. Value training works before the action encoding exercise
is complete.

## 3. Train the value network

From the repository root:

```sh
cd pylos-ml/src/main/training
```

**Run all following Cargo commands from this directory.**

In [train_value.rs](examples/train_value.rs), focus on these three settings:

- `Encoding`: the input features, currently `SphereOccupancy`. Add your own
  encodings implementing `InputEncoding` in [features.rs](src/features.rs),
  import yours in the example, and replace `SphereOccupancy` in
  `type Encoding = SphereOccupancy;`.
- `HIDDEN_SIZE`: the number of hidden units. Use a positive multiple of 32
  for export.
- `DATA_PATHS`: the collected data to train on. Set it to your actual run
  directory, for example:

```rust
const DATA_PATHS: &[&str] = &["data/run_20261109_140000_000"];
```

Use your own timestamp. Point at the directory containing `.pdat` files;
loading does not search subdirectories. Leave the other settings at their
defaults to get started. Then run:

```sh
cargo run --release --example train_value
```

Cargo downloads dependencies automatically; the first build can take several
minutes. Progress reports show loss, speed, and estimated time remaining.

A **superbatch** is a group of training batches used for progress reporting and
checkpointing. The loader repeatedly reads and shuffles the data, so a
superbatch is not one pass through the dataset.

## 4. Train the policy network

Before running [train_policy.rs](examples/train_policy.rs):

1. Complete `mapActionToActionId` in
   [StudentPlayerMCTS.java](../java/be/kuleuven/pylos/StudentPlayerMCTS.java),
   assigning consistent, distinct indices to actions.
2. Collect new data using that mapping.
3. Replace `NUM_OUTPUTS = todo!()` with your action vocabulary size and match
   Java's `PolicyNetwork.OUTPUT_SIZE`. The example will not compile until this
   TODO is completed.
4. Set the policy example's own `DATA_PATHS` to the new dataset.

```sh
cargo run --release --example train_policy
```

`NUM_OUTPUTS` is the total vocabulary size. The separate `NUM_ACTIONS = 32`
limits entries per training position: the policy loader skips empty policies
and positions with more than 32 entries.

## 5. Choose a GPU backend

The commands above use **WGPU** by default and require a compatible graphics
device and driver. CUDA is optional.

With an NVIDIA GPU, compatible driver, and
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), add `--features cuda`
to select CUDA:

```sh
cargo run --release --features cuda --example train_value
cargo run --release --features cuda --example train_policy
```

## 6. Find your trained model

Checkpoints are saved every 10 superbatches and at completion, for example in
`checkpoints/value-60-32-10/`:

- `model.bpk`: the Burn model.
- `quantised.bin`: exported parameters for the Java inference exercise.

Implement `ValueNetwork.evaluate` and `PolicyNetwork.priors` in
[StudentPlayerMCTS.java](../java/be/kuleuven/pylos/StudentPlayerMCTS.java), pointing
`WEIGHTS_PATH` at your export and matching the training dimensions and encoding.

Stopping training does not save an extra checkpoint; rerunning starts a new
model. Change `NET_NAME` between experiments to avoid replacing saved models.
