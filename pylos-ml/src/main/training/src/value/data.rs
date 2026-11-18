use crate::common::position::Position;

/// A value record with the game score from the side-to-move perspective.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValueSample {
    pub position: Position,
    /// Loss = 0, draw = 0.5, win = 1.
    pub outcome: f32,
}
