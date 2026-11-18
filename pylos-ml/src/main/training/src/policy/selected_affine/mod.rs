//! A selected policy head with CubeCL forward and backward kernels.
//!
//! Parameters remain Burn Linear tensors; only the training computation changes.
//! Two kernels evaluate legal logits and accumulate gradients directly. Devices
//! without native f32 atomic add use Burn matrix operations for backward.

mod kernel;

use burn::{
    backend::{
        Autodiff, Dispatch,
        autodiff::{
            NodeId,
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        backend_extension,
        fusion::custom::TensorSpec,
        tensor::{FloatTensor, IntTensor},
    },
    tensor::{DType, Int, Tensor},
};

#[backend_extension(Autodiff, Cube, Fusion)]
pub trait SelectedAffineBackend: burn::backend::Backend {
    #[fusion(dtype = hidden, shape = actions)]
    fn selected_affine(
        hidden: FloatTensor<Self>,
        actions: IntTensor<Self>,
        weights: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    #[fusion(meta = |hidden, _actions, weights, _grad| (
        hidden.clone(),
        weights.clone(),
        TensorSpec::new([weights.shape[0]].into(), hidden.dtype),
    ))]
    fn selected_affine_backward(
        hidden: FloatTensor<Self>,
        actions: IntTensor<Self>,
        weights: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
}

/// Evaluate the legal action slots. Weights use Linear's [hidden, outputs] layout.
/// Out-of-range action slots produce -10000 and contribute no gradients.
pub fn forward(
    hidden: Tensor<2>,
    actions: Tensor<2, Int>,
    weights: Tensor<2>,
    bias: Tensor<1>,
) -> Tensor<2> {
    let [batch, width] = hidden.dims();
    let [outputs] = bias.dims();
    assert!(batch > 0 && width > 0 && outputs > 0 && actions.dims()[1] > 0);
    assert_eq!(actions.dims()[0], batch);
    assert_eq!(weights.dims(), [width, outputs]);
    for dtype in [hidden.dtype(), weights.dtype(), bias.dtype()] {
        assert_eq!(dtype, DType::F32, "selected affine requires f32 parameters");
    }
    assert_eq!(
        actions.dtype(),
        DType::I32,
        "selected affine requires i32 indices"
    );
    // The kernel makes this small matrix contiguous for coalesced action-row reads.
    // Autodiff transposes its gradient back to Linear's original parameter layout.
    Tensor::from_dispatch(Dispatch::selected_affine(
        hidden.into_dispatch(),
        actions.into_dispatch(),
        weights.transpose().into_dispatch(),
        bias.into_dispatch(),
    ))
}

impl<B: SelectedAffineBackend, C: CheckpointStrategy> SelectedAffineBackend for Autodiff<B, C> {
    fn selected_affine(
        hidden: FloatTensor<Self>,
        actions: IntTensor<Self>,
        weights: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct SelectedAffineBackward;

        impl<B: SelectedAffineBackend> Backward<B, 3> for SelectedAffineBackward {
            type State = (NodeId, NodeId, IntTensor<B>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let (hidden, weights, actions) = ops.state;
                let hidden = checkpointer.retrieve_node_output(hidden);
                let weights = checkpointer.retrieve_node_output(weights);
                let grad = grads.consume::<B>(&ops.node);
                let (dh, dw, db) = B::selected_affine_backward(hidden, actions, weights, grad);
                for (node, gradient) in ops.parents.into_iter().zip([dh, dw, db]) {
                    if let Some(node) = node {
                        grads.register::<B>(node.id, gradient);
                    }
                }
            }
        }

        match SelectedAffineBackward
            .prepare::<C>([hidden.node(), weights.node(), bias.node()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let hidden_state = prep.checkpoint(&hidden);
                let weights_state = prep.checkpoint(&weights);
                let output = B::selected_affine(
                    hidden.into_primitive(),
                    actions.clone(),
                    weights.into_primitive(),
                    bias.into_primitive(),
                );
                prep.finish((hidden_state, weights_state, actions), output)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::selected_affine(
                hidden.into_primitive(),
                actions,
                weights.into_primitive(),
                bias.into_primitive(),
            )),
        }
    }

    fn selected_affine_backward(
        _hidden: FloatTensor<Self>,
        _actions: IntTensor<Self>,
        _weights: FloatTensor<Self>,
        _grad: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        panic!("selected affine does not support second derivatives")
    }
}
