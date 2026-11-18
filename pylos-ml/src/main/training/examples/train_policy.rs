use std::path::Path;
use burn::optim::AdamWConfig;
use burn::prelude::Device;
use pylos_train::common::loader::{DataLoader, NUM_ACTIONS};
use pylos_train::common::{logger, run, save};
use pylos_train::common::inputs::{BinaryInput, DenseInput, InputsMapper, SparseInput};
use pylos_train::common::run::{TrainingSchedule, TrainingSteps};
use pylos_train::features::{InputEncoding, SphereOccupancy};
use pylos_train::policy::data::PolicySample;
use pylos_train::policy::model::Policy;

#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Training configuration.
const NET_NAME: &str = "policy";

type Encoding = SphereOccupancy;
const HIDDEN_SIZE: usize = 32;
const NUM_OUTPUTS: usize = 991;
const DATA_PATHS: &[&str] = &["data/run_20261118_153150_692"];

const INPUT_SIZE: usize = Encoding::NUM_FEATURES;
const SUPERBATCHES: usize = 50;
const SAVE_RATE: usize = 10;
const MAPPING_THREADS: usize = 2;

pub fn main() -> anyhow::Result<()> {
    let device = Device::default().autodiff();
    let mut model = Policy::new(INPUT_SIZE, HIDDEN_SIZE, NUM_OUTPUTS, &device);
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
    let dataloader = DataLoader::for_policy(DATA_PATHS, 16 * schedule.steps.batch_size);
    let mapper = make_inputs_mapper();

    run::train(
        &mut model,
        &mut optimizer,
        &device,
        &schedule,
        &dataloader,
        &mapper,
        |model, ((inputs, actions), targets)| model.loss(inputs, actions, targets),
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

fn make_inputs_mapper() -> InputsMapper<PolicySample, ((BinaryInput, SparseInput), DenseInput<f32>)> {
    InputsMapper::new(
        ((BinaryInput::new(INPUT_SIZE), SparseInput::new(NUM_OUTPUTS, NUM_ACTIONS)), DenseInput::<f32>::new(NUM_ACTIONS)),
        MAPPING_THREADS,
        |sample: &PolicySample, ((features, actions), targets)| {
            Encoding::active_features(&sample.position, |index| {
                features[index / 32] |= (1u32 << (index % 32)) as i32;
            });
            assert!(!sample.policy.is_empty(), "policy distribution has no visits");
            assert!(sample.policy.len() <= actions.len(), "too many legal actions");
            let total: f64 = sample
                .policy
                .iter()
                .map(|&(_, weight)| f64::from(weight))
                .sum();
            assert!(total > 0.0, "policy distribution has no visits");
            for (slot, &(index, weight)) in sample.policy.iter().enumerate() {
                assert!(usize::from(index) < NUM_OUTPUTS, "output index out of range");
                actions[slot] = i32::from(index);
                targets[slot] = (f64::from(weight) / total) as f32;
            }
        },
    )
}