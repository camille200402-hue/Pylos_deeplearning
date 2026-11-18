use std::path::Path;
use burn::optim::AdamWConfig;
use burn::prelude::Device;
use pylos_train::common::loader::DataLoader;
use pylos_train::common::{logger, run, save};
use pylos_train::common::inputs::{BinaryInput, DenseInput, InputsMapper};
use pylos_train::common::run::{TrainingSchedule, TrainingSteps};
use pylos_train::features::{InputEncoding, SphereOccupancy};
use pylos_train::value::data::ValueSample;
use pylos_train::value::model::Value;

#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Training configuration.
const NET_NAME: &str = "value";

type Encoding = SphereOccupancy;
const HIDDEN_SIZE: usize = 32;
const DATA_PATHS: &[&str] = &["data/run_20261118_152058_567"];

const INPUT_SIZE: usize = Encoding::NUM_FEATURES;
const SUPERBATCHES: usize = 50;
const SAVE_RATE: usize = 10;
const MAPPING_THREADS: usize = 2;

fn main() -> anyhow::Result<()> {
    let device = Device::default().autodiff();
    let mut model = Value::new(INPUT_SIZE, HIDDEN_SIZE, &device);
    let mut optimizer = AdamWConfig::new().init();
    let schedule = TrainingSchedule {
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6_104,
            superbatches: SUPERBATCHES,
        },
        log_rate: 256,
        lr_schedule: Box::new(|step| {
            const INITIAL_LR: f64 = 1e-3;
            const FINAL_LR: f64 = 1e-6;
            let lambda = step.superbatch().saturating_sub(1) as f64 / (SUPERBATCHES - 1) as f64;
            INITIAL_LR + lambda * (FINAL_LR - INITIAL_LR)
        }),
    };
    let steps = schedule.steps;
    let dataloader = DataLoader::for_value(DATA_PATHS, 16 * steps.batch_size);
    let mapper = make_inputs_mapper();

    run::train(
        &mut model,
        &mut optimizer,
        &device,
        &schedule,
        &dataloader,
        &mapper,
        |model, (inputs, outcomes)| model.loss(inputs, outcomes),
        |step, stats| logger::report_batch(steps, step, stats),
        |model, step, stats| {
            logger::report_superbatch_finished(steps, step, stats);
            let superbatch = step.superbatch();
            if superbatch.is_multiple_of(SAVE_RATE) || superbatch == step.final_superbatch() {
                let name = format!("{NET_NAME}-{INPUT_SIZE}-{HIDDEN_SIZE}-{superbatch}");
                let directory = Path::new("checkpoints").join(name);
                save::checkpoint(model, &directory).unwrap();
                println!("Saved {}", directory.display());
            }
        },
    );
    Ok(())
}

fn make_inputs_mapper() -> InputsMapper<ValueSample, (BinaryInput, DenseInput<f32>)> {
    InputsMapper::new(
        (BinaryInput::new(INPUT_SIZE), DenseInput::<f32>::new(1)),
        MAPPING_THREADS,
        |sample: &ValueSample, (features, outcomes)| {
            Encoding::active_features(&sample.position, |index| {
                features[index / 32] |= (1u32 << (index % 32)) as i32;
            });
            outcomes[0] = sample.outcome;
        },
    )
}