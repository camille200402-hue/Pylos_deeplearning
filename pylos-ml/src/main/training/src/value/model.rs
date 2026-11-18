use crate::common::model::{CReLU, TwoLayerNetwork};
use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig,
        loss::{MseLoss, Reduction},
    },
    tensor::{Device, Tensor, activation::sigmoid},
};

/// Binary input features -> CReLU hidden layer -> side-to-move win logit.
#[derive(Module, Debug)]
pub struct Value {
    pub l0: Linear,
    activation: CReLU,
    pub l1: Linear,
}

impl TwoLayerNetwork for Value {
    fn layers(&self) -> (&Linear, &Linear) {
        (&self.l0, &self.l1)
    }

    fn layers_mut(&mut self) -> (&mut Linear, &mut Linear) {
        (&mut self.l0, &mut self.l1)
    }
}

impl Value {
    pub fn new(input_size: usize, hidden_size: usize, device: &Device) -> Self {
        Self {
            l0: LinearConfig::new(input_size, hidden_size).init(device),
            activation: CReLU,
            l1: LinearConfig::new(hidden_size, 1).init(device),
        }
    }

    pub fn forward(&self, inputs: Tensor<2>) -> Tensor<2> {
        let hidden = self.activation.forward(self.l0.forward(inputs));
        self.l1.forward(hidden)
    }

    /// Mean squared error between side-to-move win probabilities and game outcomes.
    pub fn loss(&self, inputs: Tensor<2>, outcomes: Tensor<2>) -> Tensor<1> {
        MseLoss::new().forward(sigmoid(self.forward(inputs)), outcomes, Reduction::Mean)
    }
}
