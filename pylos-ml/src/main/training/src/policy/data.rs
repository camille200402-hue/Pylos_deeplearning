use crate::common::position::Position;

/// A raw policy record, independent of the model's input representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicySample {
    pub position: Position,
    /// Ordered output IDs and unnormalized visit weights from PYLOSDAT.
    pub policy: Vec<(u16, u16)>,
}
