use super::{
    batch::{self, PrefetchStats},
    inputs::{InputLayout, InputsMapper},
    loader::DataLoader,
    logger,
    model::{TwoLayerNetwork, clip_for_quantization},
};
use burn::{
    optim::{GradientsParams, ModuleOptimizer},
    tensor::{Device, Tensor},
};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct TrainingSteps {
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub superbatches: usize,
}

pub struct TrainingSchedule {
    pub steps: TrainingSteps,
    pub log_rate: usize,
    pub lr_schedule: Box<dyn Fn(Step) -> f64>,
}

/// Identifies a batch: superbatches start at 1, batches within them start at 0.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    steps: TrainingSteps,
    batches: usize,
}

impl Step {
    pub fn new(steps: TrainingSteps) -> Self {
        assert!(steps.batch_size > 0 && steps.batches_per_superbatch > 0 && steps.superbatches > 0);
        Self { steps, batches: 0 }
    }

    pub fn batch(self) -> usize {
        self.batches % self.steps.batches_per_superbatch
    }

    pub fn superbatch(self) -> usize {
        self.batches / self.steps.batches_per_superbatch + 1
    }

    pub fn final_superbatch(self) -> usize {
        self.steps.superbatches
    }

    pub fn is_superbatch_end(self) -> bool {
        self.batch() + 1 == self.steps.batches_per_superbatch
    }

    pub fn advance(&mut self) {
        assert!(!self.finished());
        self.batches += 1;
    }

    pub fn finished(self) -> bool {
        self.batches == self.steps.superbatches * self.steps.batches_per_superbatch
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrainingStats {
    pub loss: Option<f32>,
    pub learning_rate: f64,
    pub elapsed: Duration,
    pub superbatch_elapsed: Duration,
}

pub fn train<T, M, L>(
    model: &mut M,
    optimizer: &mut ModuleOptimizer,
    device: &Device,
    schedule: &TrainingSchedule,
    dataloader: &DataLoader<T>,
    mapper: &InputsMapper<T, L>,
    loss_fn: impl Fn(&M, L::Batch) -> Tensor<1>,
    mut on_batch: impl FnMut(Step, TrainingStats),
    mut on_superbatch: impl FnMut(&M, Step, TrainingStats),
) -> PrefetchStats
where
    T: Send + Sync + 'static,
    M: TwoLayerNetwork,
    L: InputLayout,
{
    println!("Starting training...");
    let mut step = Step::new(schedule.steps);
    assert!(schedule.log_rate > 0);
    let started = Instant::now();
    let mut superbatch_started = started;
    let batch_size = schedule.steps.batch_size;
    let stats = batch::prefetch(
        device,
        mapper,
        batch_size,
        batch_size * schedule.steps.batches_per_superbatch * schedule.steps.superbatches,
        512 * 1024 * 1024,
        |read| dataloader.read_chunks(read),
        |buffer| {
            for start in (0..buffer.len()).step_by(batch_size) {
                let learning_rate = (schedule.lr_schedule)(step);
                let loss = loss_fn(model, buffer.batch(start..start + batch_size));
                let gradients = GradientsParams::from_grads(loss.backward(), model);
                device.flush().expect("flushing backward operations");
                *model = optimizer.step(learning_rate, model.clone(), gradients);
                clip_for_quantization(model);

                let loss = if step.batch() == 0
                    || (step.batch() + 1).is_multiple_of(schedule.log_rate)
                    || step.is_superbatch_end()
                {
                    let value = loss.into_scalar::<f32>();
                    assert!(value.is_finite(), "non-finite loss at {step:?}");
                    Some(value)
                } else {
                    None
                };
                let now = Instant::now();
                let stats = TrainingStats {
                    loss,
                    learning_rate,
                    elapsed: now - started,
                    superbatch_elapsed: now - superbatch_started,
                };
                on_batch(step, stats);
                if step.is_superbatch_end() {
                    on_superbatch(model, step, stats);
                    superbatch_started = Instant::now();
                }
                step.advance();
                if step.finished() {
                    return true;
                }
            }
            false
        },
    );
    assert!(
        step.finished(),
        "data loader stopped before all scheduled batches completed"
    );
    logger::report_training_finished(started.elapsed());
    stats
}
