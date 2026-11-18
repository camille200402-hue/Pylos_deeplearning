use crate::common::inputs::{GpuInputs, InputLayout, InputsMapper};
use burn::tensor::Device;
use std::{
    panic::resume_unwind,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct PrefetchStats {
    pub mapped_rows: usize,
    pub mapping_time: Duration,
    /// CPU time submitting uploads; transfer completion is asynchronous.
    pub upload_time: Duration,
    pub uploaded_bytes: usize,
    /// Largest individual GPU input buffer, counting all tuple components.
    pub peak_buffer_bytes: usize,
}

fn chunk_capacity(
    batch_size: usize,
    positions: usize,
    row_bytes: usize,
    buffer_bytes: usize,
) -> usize {
    assert!(batch_size > 0 && positions > 0 && row_bytes > 0);
    assert!(positions.is_multiple_of(batch_size));
    let batch_bytes = row_bytes
        .checked_mul(batch_size)
        .expect("GPU batch size overflow");
    let batches = (buffer_bytes / batch_bytes)
        .max(1)
        .min(positions / batch_size);
    batches * batch_size
}

/// Map and submit uploads on one worker, queuing one GPU buffer ahead of training.
pub fn prefetch<T: Send + Sync + 'static, L: InputLayout>(
    device: &Device,
    mapper: &InputsMapper<T, L>,
    batch_size: usize,
    positions: usize,
    buffer_bytes: usize,
    read: impl FnOnce(&mut dyn FnMut(&[T]) -> bool) + Send,
    mut consume: impl FnMut(&GpuInputs<L>) -> bool,
) -> PrefetchStats {
    let row_bytes = mapper.row_bytes();
    let capacity = chunk_capacity(batch_size, positions, row_bytes, buffer_bytes);
    std::thread::scope(|scope| {
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let upload_device = device.clone();
        let worker = scope.spawn(move || {
            let mut stats = PrefetchStats::default();
            let mut remaining = positions;
            let mut filled = 0;
            let mut host = None;
            read(&mut |mut samples| {
                while !samples.is_empty() && remaining > 0 {
                    let count = (capacity - filled).min(samples.len()).min(remaining);
                    let buffers = host.get_or_insert_with(|| mapper.allocate_for_mapping(capacity));
                    let started = Instant::now();
                    mapper.map_into(&samples[..count], buffers, filled..filled + count);
                    stats.mapping_time += started.elapsed();
                    stats.mapped_rows += count;
                    filled += count;
                    remaining -= count;
                    samples = &samples[count..];
                    if filled == capacity || remaining == 0 {
                        let started = Instant::now();
                        let buffer = host.take().unwrap().upload(filled, &upload_device);
                        stats.upload_time += started.elapsed();
                        let bytes = filled * row_bytes;
                        stats.uploaded_bytes += bytes;
                        stats.peak_buffer_bytes = stats.peak_buffer_bytes.max(bytes);
                        filled = 0;
                        if sender.send(buffer).is_err() {
                            return true;
                        }
                    }
                }
                remaining == 0
            });
            // Tensor handles order writes before readers across threads. Drain once
            // at shutdown instead of waiting at every upload and consumption boundary.
            upload_device.sync().expect("finishing queued GPU uploads");
            stats
        });
        while let Ok(buffer) = receiver.recv() {
            let stop = consume(&buffer);
            if stop {
                break;
            }
        }
        drop(receiver);
        // Disconnecting releases the worker if it is blocked on the full queue.
        let result = worker.join();
        device.sync().expect("finishing queued GPU computation");
        result.unwrap_or_else(|error| resume_unwind(error))
    })
}
