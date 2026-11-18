//! Typed host-buffer mapping and direct tensor uploads, independent of the model.

mod buffer;

use buffer::BufferPool;
use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use std::{iter::Zip, ops::Range, slice::ChunksMut, sync::Arc, thread};

/// One input, or recursively nested pairs, with flat host storage and typed row views.
pub trait InputLayout: Clone + PartialEq + Send + Sync + 'static {
    type Host: Send;
    type View<'a>: Send;
    type Row<'a>: Send;
    type Chunks<'a>: Iterator<Item = Self::View<'a>> + Send;
    type Rows<'a>: Iterator<Item = Self::Row<'a>> + Send;
    /// Storage retained for a complete uploaded chunk.
    type Gpu: Send;
    /// Views or decoded tensors consumed by the model for one minibatch.
    type Batch;

    fn row_bytes(&self) -> usize;
    fn allocate(&self, rows: usize) -> Self::Host;
    /// Scratch storage; the caller must map every uploaded row before exposing it.
    fn allocate_uncleared(&self, rows: usize) -> Self::Host {
        self.allocate(rows)
    }
    fn view<'a>(&self, host: &'a mut Self::Host, rows: Range<usize>) -> Self::View<'a>;
    fn clear(&self, view: &mut Self::View<'_>);
    fn chunks<'a>(&self, view: Self::View<'a>, rows: usize) -> Self::Chunks<'a>;
    fn rows<'a>(&self, view: Self::View<'a>) -> Self::Rows<'a>;
    fn validate(&self, _host: &Self::Host, _rows: Range<usize>) {}
    fn upload(&self, host: Self::Host, rows: usize, device: &Device) -> Self::Gpu;
    fn batch(&self, gpu: &Self::Gpu, rows: Range<usize>) -> Self::Batch;
}

#[derive(Clone, Debug)]
pub struct DenseInput<T> {
    width: usize,
    pool: Arc<BufferPool<T>>,
}

impl<T> PartialEq for DenseInput<T> {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
    }
}

impl<T> DenseInput<T> {
    pub fn new(width: usize) -> Self {
        assert!(width > 0);
        Self {
            width,
            pool: Arc::default(),
        }
    }
}

macro_rules! dense_input {
    ($element:ty, $kind:ty, $dtype:expr, $zero:expr) => {
        impl InputLayout for DenseInput<$element> {
            type Host = Vec<$element>;
            type View<'a> = &'a mut [$element];
            type Row<'a> = &'a mut [$element];
            type Chunks<'a> = ChunksMut<'a, $element>;
            type Rows<'a> = ChunksMut<'a, $element>;
            type Gpu = Tensor<2, $kind>;
            type Batch = Self::Gpu;

            fn row_bytes(&self) -> usize {
                self.width * size_of::<$element>()
            }
            fn allocate(&self, rows: usize) -> Self::Host {
                self.pool.take(rows * self.width, $zero)
            }
            fn allocate_uncleared(&self, rows: usize) -> Self::Host {
                self.pool.take_uncleared(rows * self.width, $zero)
            }
            fn view<'a>(&self, host: &'a mut Self::Host, rows: Range<usize>) -> Self::View<'a> {
                &mut host[rows.start * self.width..rows.end * self.width]
            }
            fn clear(&self, view: &mut Self::View<'_>) {
                view.fill($zero);
            }
            fn chunks<'a>(&self, view: Self::View<'a>, rows: usize) -> Self::Chunks<'a> {
                view.chunks_mut(rows * self.width)
            }
            fn rows<'a>(&self, view: Self::View<'a>) -> Self::Rows<'a> {
                view.chunks_mut(self.width)
            }
            fn upload(&self, host: Self::Host, rows: usize, device: &Device) -> Self::Gpu {
                let bytes = self.pool.share(host, rows * self.width);
                Tensor::from_data(
                    TensorData::from_bytes(bytes, [rows, self.width], $dtype),
                    device,
                )
            }
            fn batch(&self, gpu: &Self::Gpu, rows: Range<usize>) -> Self::Gpu {
                gpu.clone().slice([rows, 0..self.width])
            }
        }
    };
}

dense_input!(f32, burn::tensor::Float, DType::F32, 0.0);
dense_input!(i32, Int, DType::I32, 0);

/// Packed binary features, expanded to dense f32 only for the current minibatch.
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryInput {
    num_features: usize,
    words: DenseInput<i32>,
}

impl BinaryInput {
    pub fn new(num_features: usize) -> Self {
        assert!(num_features > 0);
        Self {
            num_features,
            words: DenseInput::new(num_features.div_ceil(32)),
        }
    }
}

impl InputLayout for BinaryInput {
    type Host = Vec<i32>;
    type View<'a> = &'a mut [i32];
    type Row<'a> = &'a mut [i32];
    type Chunks<'a> = ChunksMut<'a, i32>;
    type Rows<'a> = ChunksMut<'a, i32>;
    type Gpu = (Tensor<2, Int>, Tensor<3, Int>);
    type Batch = Tensor<2>;

    fn row_bytes(&self) -> usize {
        self.words.row_bytes()
    }
    fn allocate(&self, rows: usize) -> Self::Host {
        self.words.allocate(rows)
    }
    fn allocate_uncleared(&self, rows: usize) -> Self::Host {
        self.words.allocate_uncleared(rows)
    }
    fn view<'a>(&self, host: &'a mut Self::Host, rows: Range<usize>) -> Self::View<'a> {
        self.words.view(host, rows)
    }
    fn clear(&self, view: &mut Self::View<'_>) {
        self.words.clear(view);
    }
    fn chunks<'a>(&self, view: Self::View<'a>, rows: usize) -> Self::Chunks<'a> {
        self.words.chunks(view, rows)
    }
    fn rows<'a>(&self, view: Self::View<'a>) -> Self::Rows<'a> {
        self.words.rows(view)
    }
    fn validate(&self, host: &Self::Host, rows: Range<usize>) {
        let tail = self.num_features % 32;
        if tail != 0 {
            assert!(
                host[rows.start * self.words.width..rows.end * self.words.width]
                    .chunks(self.words.width)
                    .all(|words| (words[self.words.width - 1] as u32) >> tail == 0),
                "binary feature index out of range"
            );
        }
    }
    fn upload(&self, host: Self::Host, rows: usize, device: &Device) -> Self::Gpu {
        (
            self.words.upload(host, rows, device),
            Tensor::<1, Int>::arange(0..32, device).reshape([1, 1, 32]),
        )
    }
    fn batch(&self, gpu: &Self::Gpu, rows: Range<usize>) -> Self::Batch {
        let size = rows.len();
        let words = self.words.batch(&gpu.0, rows).unsqueeze_dim::<3>(2);
        words
            .bitwise_right_shift(gpu.1.clone())
            .bitwise_and_scalar(1)
            .float()
            .reshape([size, self.words.width * 32])
            .slice([0..size, 0..self.num_features])
    }
}

/// Binary feature indices; unused slots contain -1. Index zero is valid.
#[derive(Clone, Debug)]
pub struct SparseInput {
    num_features: usize,
    max_active: usize,
    pool: Arc<BufferPool<i32>>,
}

impl PartialEq for SparseInput {
    fn eq(&self, other: &Self) -> bool {
        self.num_features == other.num_features && self.max_active == other.max_active
    }
}

impl Eq for SparseInput {}

impl SparseInput {
    pub fn new(num_features: usize, max_active: usize) -> Self {
        assert!(num_features > 0 && num_features <= i32::MAX as usize);
        assert!(max_active > 0);
        Self {
            num_features,
            max_active,
            pool: Arc::default(),
        }
    }
}

impl InputLayout for SparseInput {
    type Host = Vec<i32>;
    type View<'a> = &'a mut [i32];
    type Row<'a> = &'a mut [i32];
    type Chunks<'a> = ChunksMut<'a, i32>;
    type Rows<'a> = ChunksMut<'a, i32>;
    type Gpu = Tensor<2, Int>;
    type Batch = Self::Gpu;

    fn row_bytes(&self) -> usize {
        self.max_active * size_of::<i32>()
    }
    fn allocate(&self, rows: usize) -> Self::Host {
        self.pool.take(rows * self.max_active, -1)
    }
    fn allocate_uncleared(&self, rows: usize) -> Self::Host {
        self.pool.take_uncleared(rows * self.max_active, -1)
    }
    fn view<'a>(&self, host: &'a mut Self::Host, rows: Range<usize>) -> Self::View<'a> {
        &mut host[rows.start * self.max_active..rows.end * self.max_active]
    }
    fn clear(&self, view: &mut Self::View<'_>) {
        view.fill(-1);
    }
    fn chunks<'a>(&self, view: Self::View<'a>, rows: usize) -> Self::Chunks<'a> {
        view.chunks_mut(rows * self.max_active)
    }
    fn rows<'a>(&self, view: Self::View<'a>) -> Self::Rows<'a> {
        view.chunks_mut(self.max_active)
    }
    fn validate(&self, host: &Self::Host, rows: Range<usize>) {
        assert!(
            host[rows.start * self.max_active..rows.end * self.max_active]
                .iter()
                .all(|&index| index >= -1 && index < self.num_features as i32),
            "sparse index out of range"
        );
    }
    fn upload(&self, host: Self::Host, rows: usize, device: &Device) -> Self::Gpu {
        let bytes = self.pool.share(host, rows * self.max_active);
        Tensor::from_data(
            TensorData::from_bytes(bytes, [rows, self.max_active], DType::I32),
            device,
        )
    }
    fn batch(&self, gpu: &Self::Gpu, rows: Range<usize>) -> Self::Gpu {
        gpu.clone().slice([rows, 0..self.max_active])
    }
}

/// Compose layouts recursively; larger input groups are nested pairs.
impl<A: InputLayout, B: InputLayout> InputLayout for (A, B) {
    type Host = (A::Host, B::Host);
    type View<'a> = (A::View<'a>, B::View<'a>);
    type Row<'a> = (A::Row<'a>, B::Row<'a>);
    type Chunks<'a> = Zip<A::Chunks<'a>, B::Chunks<'a>>;
    type Rows<'a> = Zip<A::Rows<'a>, B::Rows<'a>>;
    type Gpu = (A::Gpu, B::Gpu);
    type Batch = (A::Batch, B::Batch);

    fn row_bytes(&self) -> usize {
        self.0.row_bytes() + self.1.row_bytes()
    }

    fn allocate(&self, rows: usize) -> Self::Host {
        (self.0.allocate(rows), self.1.allocate(rows))
    }
    fn allocate_uncleared(&self, rows: usize) -> Self::Host {
        (
            self.0.allocate_uncleared(rows),
            self.1.allocate_uncleared(rows),
        )
    }

    fn view<'a>(&self, host: &'a mut Self::Host, rows: Range<usize>) -> Self::View<'a> {
        (
            self.0.view(&mut host.0, rows.clone()),
            self.1.view(&mut host.1, rows),
        )
    }

    fn clear(&self, view: &mut Self::View<'_>) {
        self.0.clear(&mut view.0);
        self.1.clear(&mut view.1);
    }

    fn chunks<'a>(&self, view: Self::View<'a>, rows: usize) -> Self::Chunks<'a> {
        self.0.chunks(view.0, rows).zip(self.1.chunks(view.1, rows))
    }

    fn rows<'a>(&self, view: Self::View<'a>) -> Self::Rows<'a> {
        self.0.rows(view.0).zip(self.1.rows(view.1))
    }

    fn validate(&self, host: &Self::Host, rows: Range<usize>) {
        self.0.validate(&host.0, rows.clone());
        self.1.validate(&host.1, rows);
    }

    fn upload(&self, host: Self::Host, rows: usize, device: &Device) -> Self::Gpu {
        (
            self.0.upload(host.0, rows, device),
            self.1.upload(host.1, rows, device),
        )
    }

    fn batch(&self, gpu: &Self::Gpu, rows: Range<usize>) -> Self::Batch {
        (
            self.0.batch(&gpu.0, rows.clone()),
            self.1.batch(&gpu.1, rows),
        )
    }
}

/// Flat, preallocated CPU buffers. Mapping can fill any range of rows repeatedly.
pub struct HostInputs<L: InputLayout> {
    layout: L,
    buffers: L::Host,
    rows: usize,
}

impl<L: InputLayout> HostInputs<L> {
    pub fn buffers(&self) -> &L::Host {
        &self.buffers
    }

    pub fn upload(self, rows: usize, device: &Device) -> GpuInputs<L> {
        assert!(rows > 0 && rows <= self.rows);
        GpuInputs {
            tensors: self.layout.upload(self.buffers, rows, device),
            layout: self.layout,
            rows,
        }
    }
}

/// Typed upload storage, sliced or decoded into minibatches by its layout.
pub struct GpuInputs<L: InputLayout> {
    layout: L,
    tensors: L::Gpu,
    rows: usize,
}

impl<L: InputLayout> GpuInputs<L> {
    pub fn len(&self) -> usize {
        self.rows
    }
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }
    pub fn batch(&self, rows: Range<usize>) -> L::Batch {
        assert!(rows.start < rows.end && rows.end <= self.rows);
        self.layout.batch(&self.tensors, rows)
    }
}

pub struct InputsMapper<T, L: InputLayout> {
    layout: L,
    threads: usize,
    #[allow(clippy::type_complexity)]
    map_batch: Arc<dyn for<'a> Fn(&[T], L::View<'a>, usize) + Send + Sync>,
}

impl<T, L: InputLayout> Clone for InputsMapper<T, L> {
    fn clone(&self) -> Self {
        Self {
            layout: self.layout.clone(),
            threads: self.threads,
            map_batch: Arc::clone(&self.map_batch),
        }
    }
}

impl<T: Send + Sync + 'static, L: InputLayout> InputsMapper<T, L> {
    pub fn new<F>(layout: L, threads: usize, map_row: F) -> Self
    where
        F: for<'a> Fn(&T, L::Row<'a>) + Send + Sync + 'static,
    {
        assert!(threads > 0);
        let inputs = layout.clone();
        let map_batch = move |samples: &[T], mut view: L::View<'_>, threads: usize| {
            inputs.clear(&mut view);
            if samples.is_empty() {
                return;
            }
            let chunk_size = samples.len().div_ceil(threads);
            thread::scope(|scope| {
                for (samples, view) in samples
                    .chunks(chunk_size)
                    .zip(inputs.chunks(view, chunk_size))
                {
                    let inputs = &inputs;
                    let map_row = &map_row;
                    scope.spawn(move || {
                        for (sample, row) in samples.iter().zip(inputs.rows(view)) {
                            map_row(sample, row);
                        }
                    });
                }
            });
        };
        Self {
            layout,
            threads,
            map_batch: Arc::new(map_batch),
        }
    }

    pub fn row_bytes(&self) -> usize {
        self.layout.row_bytes()
    }

    pub fn allocate(&self, rows: usize) -> HostInputs<L> {
        HostInputs {
            buffers: self.layout.allocate(rows),
            layout: self.layout.clone(),
            rows,
        }
    }

    pub(crate) fn allocate_for_mapping(&self, rows: usize) -> HostInputs<L> {
        HostInputs {
            buffers: self.layout.allocate_uncleared(rows),
            layout: self.layout.clone(),
            rows,
        }
    }

    pub fn map_into(&self, samples: &[T], host: &mut HostInputs<L>, rows: Range<usize>) {
        assert!(self.layout == host.layout, "input layout mismatch");
        assert!(rows.end <= host.rows);
        assert_eq!(samples.len(), rows.len());
        (self.map_batch)(
            samples,
            self.layout.view(&mut host.buffers, rows.clone()),
            self.threads,
        );
        self.layout.validate(&host.buffers, rows);
    }
}
