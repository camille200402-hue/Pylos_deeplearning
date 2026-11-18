use arrayvec::ArrayVec;
use burn::tensor::{AllocationProperty, Bytes};
use bytemuck::Pod;
use std::sync::{Arc, Mutex, Weak};

/// Recycles host vectors when the runtime releases its last upload reference.
#[derive(Debug)]
pub(super) struct BufferPool<T> {
    free: Mutex<ArrayVec<Vec<T>, 2>>,
}

impl<T> Default for BufferPool<T> {
    fn default() -> Self {
        Self {
            free: Mutex::new(ArrayVec::new()),
        }
    }
}

impl<T: Pod + Send + Sync + 'static> BufferPool<T> {
    pub(super) fn take(&self, len: usize, zero: T) -> Vec<T> {
        let mut buffer = self.take_uncleared(len, zero);
        buffer.fill(zero);
        buffer
    }

    pub(super) fn take_uncleared(&self, len: usize, zero: T) -> Vec<T> {
        let mut free = self.free.lock().unwrap();
        let reused = free
            .iter()
            .position(|buffer| buffer.capacity() >= len)
            .map(|index| free.swap_remove(index));
        drop(free);
        match reused {
            Some(mut buffer) => {
                buffer.resize(len, zero);
                buffer
            }
            None => vec![zero; len],
        }
    }

    pub(super) fn share(self: &Arc<Self>, mut buffer: Vec<T>, len: usize) -> Bytes {
        buffer.truncate(len);
        let owner = UploadBuffer {
            buffer,
            pool: Arc::downgrade(self),
        };
        Bytes::from_shared(bytes::Bytes::from_owner(owner), AllocationProperty::Native)
    }
}

struct UploadBuffer<T> {
    buffer: Vec<T>,
    pool: Weak<BufferPool<T>>,
}

impl<T: Pod> AsRef<[u8]> for UploadBuffer<T> {
    fn as_ref(&self) -> &[u8] {
        bytemuck::cast_slice(&self.buffer)
    }
}

impl<T> Drop for UploadBuffer<T> {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.upgrade() {
            let mut free = pool.free.lock().unwrap();
            // Keep only a bounded number of idle allocations for each input.
            if !free.is_full() {
                free.push(std::mem::take(&mut self.buffer));
            }
        }
    }
}
