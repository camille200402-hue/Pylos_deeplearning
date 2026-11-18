use crate::common::position::Position;

pub trait InputEncoding {
    const NUM_FEATURES: usize;

    fn active_features(position: &Position, activate: impl FnMut(usize));
}

/// Thirty side-to-move sphere locations followed by thirty opponent sphere locations.
pub struct SphereOccupancy;

impl InputEncoding for SphereOccupancy {
    const NUM_FEATURES: usize = 60;

    fn active_features(position: &Position, mut activate: impl FnMut(usize)) {
        let stm = position.color;
        for (color, offset) in [(stm, 0), (!stm, 30)] {
            let mut spheres = position.bitboards[color as usize];
            while spheres != 0 {
                activate(offset + spheres.trailing_zeros() as usize);
                spheres &= spheres - 1;
            }
        }
    }
}
