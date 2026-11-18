use burn::{
    module::{Module, Param},
    nn::Linear,
    tensor::Tensor,
};

pub const QA: i16 = 255;
pub const QB: i16 = 64;
pub(crate) const OUTPUT_BIAS_SCALE: i16 = QA * QB;

#[derive(Module, Debug, Default)]
pub(crate) struct CReLU;

impl CReLU {
    pub(crate) fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        input.clamp(0.0, 1.0)
    }
}

/// The shared parameter layout of the policy and value NNUE networks.
pub trait TwoLayerNetwork: Module {
    fn layers(&self) -> (&Linear, &Linear);
    fn layers_mut(&mut self) -> (&mut Linear, &mut Linear);
}

pub(crate) fn clip_for_quantization(model: &mut impl TwoLayerNetwork) {
    let (l0, l1) = model.layers_mut();
    clip_layer(l0, QA, QA);
    clip_layer(l1, QB, OUTPUT_BIAS_SCALE);
}

fn clip_layer(layer: &mut Linear, weight_scale: i16, bias_scale: i16) {
    layer.weight = clip_parameter(layer.weight.clone(), weight_scale);
    if let Some(bias) = &mut layer.bias {
        *bias = clip_parameter(bias.clone(), bias_scale);
    }
}

fn clip_parameter<const D: usize>(param: Param<Tensor<D>>, scale: i16) -> Param<Tensor<D>> {
    let min = f32::from(i16::MIN) / f32::from(scale);
    let max = f32::from(i16::MAX) / f32::from(scale);
    let id = param.id;
    param.map(|tensor| {
        let Some(strategy) = tensor.gradient_checkpointing_strategy() else {
            return tensor.clamp(min, max);
        };
        let require_grad = tensor.is_require_grad();
        let distributed = tensor.is_distributed();
        // The inner backend can fuse clamp with AdamW's elementwise operations.
        let clipped = Tensor::from_inner(tensor.inner().clamp(min, max))
            .with_gradient_checkpointing_strategy(strategy)
            .set_require_grad(require_grad);
        if distributed {
            clipped.set_distributed(id)
        } else {
            clipped
        }
    })
}
