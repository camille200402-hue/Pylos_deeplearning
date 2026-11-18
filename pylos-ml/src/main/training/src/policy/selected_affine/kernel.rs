use super::SelectedAffineBackend;
use burn::backend::{
    TensorMetadata,
    cubecl::dtype_to_elem_type,
    ops::{FloatTensorOps, IntTensorOps},
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::{BoolDType, IndexingUpdateOp};
use burn_cubecl::cubecl::{
    self,
    features::{AtomicUsage, Plane},
    ir::Type,
    prelude::*,
};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};

// One subgroup owns a position. Hidden features stay in registers across actions.
#[cube(launch)]
fn selected_forward(
    hidden: &Tensor<f32>,
    actions: &Tensor<i32>,
    weights: &Tensor<f32>,
    bias: &Tensor<f32>,
    output: &mut Tensor<f32>,
    #[comptime] width: usize,
    #[comptime] slots: usize,
    #[comptime] lanes: usize,
) {
    let row = CUBE_POS;
    if row >= hidden.shape(0) {
        terminate!();
    }
    let lane = UNIT_POS as usize;
    let mut features = Array::<f32>::new(width.div_ceil(lanes));
    #[unroll]
    for chunk in 0..width.div_ceil(lanes) {
        let h = lane + chunk * lanes;
        features[chunk] = 0.0;
        if h < width {
            features[chunk] = hidden[row * width + h];
        }
    }
    for slot in 0..slots {
        let index = row * slots + slot;
        let action = actions[index];
        let mut sum = f32::new(-10000.0_f32);
        if action >= 0 && (action as usize) < bias.len() {
            let mut dot = f32::new(0.0_f32);
            #[unroll]
            for chunk in 0..width.div_ceil(lanes) {
                let h = lane + chunk * lanes;
                if h < width {
                    dot += features[chunk] * weights[action as usize * width + h];
                }
            }
            sum = plane_sum(dot) + bias[action as usize];
        }
        if lane == 0 {
            output[index] = sum;
        }
    }
}

// Each subgroup owns one hidden-gradient row; parameter gradients are shared.
// Zero CReLU activations need no atomic update.
#[cube(launch)]
fn selected_backward(
    hidden: &Tensor<f32>,
    actions: &Tensor<i32>,
    weights: &Tensor<f32>,
    grad: &Tensor<f32>,
    hidden_grad: &mut Tensor<f32>,
    weights_grad: &mut Tensor<Atomic<f32>>,
    bias_grad: &mut Tensor<Atomic<f32>>,
    #[comptime] width: usize,
    #[comptime] slots: usize,
    #[comptime] lanes: usize,
    #[comptime] outputs: usize,
) {
    let row = CUBE_POS;
    if row >= hidden.shape(0) {
        terminate!();
    }
    let lane = UNIT_POS as usize;
    let mut features = Array::<f32>::new(width.div_ceil(lanes));
    let mut sums = Array::<f32>::new(width.div_ceil(lanes));
    #[unroll]
    for chunk in 0..width.div_ceil(lanes) {
        let h = lane + chunk * lanes;
        features[chunk] = 0.0;
        sums[chunk] = 0.0;
        if h < width {
            features[chunk] = hidden[row * width + h];
        }
    }
    for slot in 0..slots {
        let index = row * slots + slot;
        let action = actions[index];
        if action >= 0 && (action as usize) < outputs {
            let delta = grad[index];
            #[unroll]
            for chunk in 0..width.div_ceil(lanes) {
                let h = lane + chunk * lanes;
                if h < width {
                    let wi = action as usize * width + h;
                    sums[chunk] += delta * weights[wi];
                    let contribution = delta * features[chunk];
                    if contribution != 0.0 {
                        weights_grad[wi].fetch_add(contribution);
                    }
                }
            }
            if lane == 0 && delta != 0.0 {
                bias_grad[action as usize].fetch_add(delta);
            }
        }
    }
    #[unroll]
    for chunk in 0..width.div_ceil(lanes) {
        let h = lane + chunk * lanes;
        if h < width {
            hidden_grad[row * width + h] = sums[chunk];
        }
    }
}

fn empty_like(input: &CubeTensor) -> CubeTensor {
    let shape = input.shape();
    let handle = input
        .client
        .empty(shape.num_elements() * input.dtype.size());
    CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape,
        handle,
        input.dtype,
    )
}

fn lanes(hidden: &CubeTensor) -> usize {
    let properties = hidden.client.properties();
    assert!(
        properties.features.plane.contains(Plane::Ops),
        "selected affine requires subgroup operations"
    );
    // A workgroup no larger than the minimum subgroup size has exactly one active subgroup.
    (properties.hardware.plane_size_min as usize).clamp(1, 32)
}

// Portable alternative: selected forward, ordinary GEMMs for backward.
// Burn's scatter handles devices without native floating-point atomic add.
pub(super) fn matmul_backward(
    hidden: CubeTensor,
    actions: CubeTensor,
    weights: CubeTensor,
    grad: CubeTensor,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    let batch = hidden.shape()[0];
    let outputs = weights.shape()[0];
    let negative = CubeBackend::int_lower_elem(actions.clone(), 0.into(), BoolDType::U32);
    let too_large = CubeBackend::int_greater_equal_elem(
        actions.clone(),
        (outputs as i64).into(),
        BoolDType::U32,
    );
    let grad = CubeBackend::float_mask_fill(grad, negative, 0.0.into());
    let grad = CubeBackend::float_mask_fill(grad, too_large, 0.0.into());
    let indices = CubeBackend::int_clamp(actions, 0.into(), ((outputs - 1) as i64).into());
    let dense =
        CubeBackend::float_zeros([batch, outputs].into(), &hidden.device, hidden.dtype.into());
    let dense = CubeBackend::float_scatter(1, dense, indices, grad, IndexingUpdateOp::Add);
    let dh = CubeBackend::float_matmul(dense.clone(), weights);
    let dw = CubeBackend::float_matmul(CubeBackend::float_transpose(dense.clone()), hidden);
    let db = CubeBackend::float_sum_dim(dense, 0);
    (dh, dw, CubeBackend::float_reshape(db, [outputs].into()))
}

impl SelectedAffineBackend for CubeBackend {
    fn selected_affine(
        hidden: FloatTensor<Self>,
        actions: IntTensor<Self>,
        weights: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        hidden.assert_is_on_same_device(&actions);
        hidden.assert_is_on_same_device(&weights);
        hidden.assert_is_on_same_device(&bias);
        let hidden = into_contiguous(hidden);
        let actions = into_contiguous(actions);
        let weights = into_contiguous(weights);
        let bias = into_contiguous(bias);
        let [batch, width] = hidden.shape().dims();
        let slots = actions.shape()[1];
        let lanes = lanes(&hidden);
        let handle = hidden.client.empty(batch * slots * 4);
        let output = CubeTensor::new_contiguous(
            hidden.client.clone(),
            hidden.device.clone(),
            [batch, slots].into(),
            handle,
            hidden.dtype,
        );
        selected_forward::launch(
            &output.client,
            // One workgroup per row, spread across axes within the device limits.
            cubecl::calculate_cube_count_elemwise(&output.client, batch, CubeDim::new_1d(1)),
            CubeDim::new_1d(lanes as u32),
            hidden.into_tensor_arg(),
            actions.into_tensor_arg(),
            weights.into_tensor_arg(),
            bias.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            width,
            slots,
            lanes,
        );
        output
    }

    fn selected_affine_backward(
        hidden: FloatTensor<Self>,
        actions: IntTensor<Self>,
        weights: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let native = hidden
            .client
            .properties()
            .atomic_type_usage(Type::atomic(dtype_to_elem_type(hidden.dtype)))
            .contains(AtomicUsage::Add);
        if !native {
            return matmul_backward(hidden, actions, weights, grad);
        }
        let hidden = into_contiguous(hidden);
        let actions = into_contiguous(actions);
        let weights = into_contiguous(weights);
        let grad = into_contiguous(grad);
        let [batch, width] = hidden.shape().dims();
        let slots = actions.shape()[1];
        let outputs = weights.shape()[0];
        let lanes = lanes(&hidden);
        // Kernels use flat offsets; CUDA's default allocation can have pitched rows.
        let dw = into_contiguous(Self::float_zeros(
            weights.shape(),
            &hidden.device,
            hidden.dtype.into(),
        ));
        let db = into_contiguous(Self::float_zeros(
            [outputs].into(),
            &hidden.device,
            hidden.dtype.into(),
        ));
        let dh = empty_like(&hidden);
        selected_backward::launch(
            &dh.client,
            cubecl::calculate_cube_count_elemwise(&dh.client, batch, CubeDim::new_1d(1)),
            CubeDim::new_1d(lanes as u32),
            hidden.into_tensor_arg(),
            actions.into_tensor_arg(),
            weights.into_tensor_arg(),
            grad.into_tensor_arg(),
            dh.clone().into_tensor_arg(),
            dw.clone().into_tensor_arg(),
            db.clone().into_tensor_arg(),
            width,
            slots,
            lanes,
            outputs,
        );
        (dh, dw, db)
    }
}
