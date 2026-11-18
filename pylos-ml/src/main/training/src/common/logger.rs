use crate::common::run::{Step, TrainingStats, TrainingSteps};
use std::{
    io::{self, IsTerminal, Write},
    time::Duration,
};

pub fn report_batch(steps: TrainingSteps, step: Step, stats: TrainingStats) {
    if stats.loss.is_none() || step.is_superbatch_end() || !io::stdout().is_terminal() {
        return;
    }

    let batches = step.batch() + 1;
    let progress = batches as f64 / steps.batches_per_superbatch as f64;
    let eta = stats
        .superbatch_elapsed
        .mul_f64((steps.batches_per_superbatch - batches) as f64 / batches as f64);
    print!(
        "\r\x1b[2KSuperbatch {}/{} | lr {:.3e} | {:6.2}% | ETA {} | total ETA {}",
        step.superbatch(),
        steps.superbatches,
        stats.learning_rate,
        progress * 100.0,
        format_duration(eta),
        training_eta(steps, step, stats),
    );
    io::stdout().flush().unwrap();
}

pub fn report_superbatch_finished(steps: TrainingSteps, step: Step, stats: TrainingStats) {
    if io::stdout().is_terminal() {
        print!("\r\x1b[2K");
    }
    let positions = steps.batch_size * steps.batches_per_superbatch;
    let throughput = positions as f64 / stats.superbatch_elapsed.as_secs_f64();
    println!(
        "Superbatch {}/{} complete | lr {:.3e} | last batch loss {:.6} | {} | {throughput:.0} pos/s | training ETA {}",
        step.superbatch(),
        steps.superbatches,
        stats.learning_rate,
        stats.loss.unwrap(),
        format_duration(stats.superbatch_elapsed),
        training_eta(steps, step, stats),
    );
}

pub fn report_training_finished(elapsed: Duration) {
    println!(
        "Training complete | total time {}",
        format_duration(elapsed)
    );
}

fn training_eta(steps: TrainingSteps, step: Step, stats: TrainingStats) -> String {
    let completed = (step.superbatch() - 1) as f64
        + (step.batch() + 1) as f64 / steps.batches_per_superbatch as f64;
    if completed < 1.0 {
        "estimating...".to_owned()
    } else {
        format_duration(
            stats
                .elapsed
                .mul_f64((steps.superbatches as f64 - completed) / completed),
        )
    }
}

fn format_duration(duration: Duration) -> String {
    let tenths = (duration.as_millis() + 50) / 100;
    let hours = tenths / 36_000;
    let minutes = tenths / 600 % 60;
    let seconds = (tenths % 600) as f64 / 10.0;
    if hours > 0 {
        format!("{hours}h {minutes}m {seconds:.1}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds:.1}s")
    } else {
        format!("{seconds:.1}s")
    }
}
