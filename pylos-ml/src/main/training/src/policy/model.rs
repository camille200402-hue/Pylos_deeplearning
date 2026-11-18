use super::selected_affine;
use crate::common::model::{CReLU, TwoLayerNetwork};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Device, Int, Tensor, activation::log_softmax},
};

/// Binary input features -> CReLU hidden layer -> action logits.
#[derive(Module, Debug)]
pub struct Policy {
    pub l0: Linear,
    activation: CReLU,
    pub l1: Linear,
}

impl TwoLayerNetwork for Policy {
    fn layers(&self) -> (&Linear, &Linear) {
        (&self.l0, &self.l1)
    }

    fn layers_mut(&mut self) -> (&mut Linear, &mut Linear) {
        (&mut self.l0, &mut self.l1)
    }
}

impl Policy {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, device: &Device) -> Self {
        Self {
            l0: LinearConfig::new(input_size, hidden_size).init(device),
            activation: CReLU,
            l1: LinearConfig::new(hidden_size, output_size).init(device),
        }
    }

    /// Full logits for inference.
    #[allow(dead_code)] // Training evaluates selected logits only.
    pub fn forward(&self, inputs: Tensor<2>) -> Tensor<2> {
        let hidden = self.activation.forward(self.l0.forward(inputs));
        self.l1.forward(hidden)
    }

    /// Cross-entropy against normalized visits, averaged over positions.
    pub fn loss(
        &self,
        inputs: Tensor<2>,
        actions: Tensor<2, Int>,
        targets: Tensor<2>,
    ) -> Tensor<1> {
        assert_eq!(
            actions.dims(),
            targets.dims(),
            "policy targets must match action slots"
        );
        let hidden = self.activation.forward(self.l0.forward(inputs));
        let logits = selected_affine::forward(
            hidden,
            actions,
            self.l1.weight.val(),
            self.l1.bias.as_ref().unwrap().val(),
        );
        -(targets * log_softmax(logits, 1)).sum_dim(1).mean()
    }
}
