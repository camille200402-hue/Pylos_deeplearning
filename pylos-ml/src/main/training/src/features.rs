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
const fn loc_bit(z: usize, x: usize, y: usize) -> u32 {
    let offset = match z {
        0 => 0,
        1 => 16,
        2 => 25,
        3 => 29,
        _ => unreachable!(),
    };
    let width = 4 - z;
    1u32 << (offset + y * width + x)
}

/// 14 possible 2x2 squares (9 on z=0, 4 on z=1, 1 on z=2).
const SQUARE_MASKS: [u32; 14] = {
    let mut masks = [0u32; 14];
    let mut idx = 0;
    let mut z = 0;
    while z < 3 {
        let limit = 3 - z;
        let mut y = 0;
        while y < limit {
            let mut x = 0;
            while x < limit {
                masks[idx] = loc_bit(z, x, y)
                    | loc_bit(z, x + 1, y)
                    | loc_bit(z, x, y + 1)
                    | loc_bit(z, x + 1, y + 1);
                idx += 1;
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }
    masks
};

/// Bitmask of the 4 supporting slots on z-1 for each location (0 on z=0).
const SUPPORT_MASKS: [u32; 30] = {
    let mut masks = [0u32; 30];
    let mut z = 1;
    while z < 4 {
        let offset = match z {
            1 => 16,
            2 => 25,
            3 => 29,
            _ => unreachable!(),
        };
        let width = 4 - z;
        let mut y = 0;
        while y < width {
            let mut x = 0;
            while x < width {
                let idx = offset + y * width + x;
                masks[idx] = loc_bit(z - 1, x, y)
                    | loc_bit(z - 1, x + 1, y)
                    | loc_bit(z - 1, x, y + 1)
                    | loc_bit(z - 1, x + 1, y + 1);
                x += 1;
            }
            y += 1;
        }
        z += 1;
    }
    masks
};

/// Bitmask of upper slots on z+1 resting on top of each location.
const ABOVE_MASKS: [u32; 30] = {
    let mut above = [0u32; 30];
    let mut upper = 16;
    while upper < 30 {
        let sup = SUPPORT_MASKS[upper];
        let mut lower = 0;
        while lower < 30 {
            if (sup & (1u32 << lower)) != 0 {
                above[lower] |= 1u32 << upper;
            }
            lower += 1;
        }
        upper += 1;
    }
    above
};

/// 162-feature tactical encoding:
///   0..60   : SphereOccupancy (30 stm + 30 opp)
///   60..102 : 14 2x2 squares x 3 features (stm 3-of-4 threat, opp 3-of-4 threat, stm 4-of-4)
///   102..132: 30 placeable locations (empty & all 4 supports present)
///   132..162: 30 unblocked locations (occupied & no sphere resting above)
pub struct TacticalEncoding;

impl InputEncoding for TacticalEncoding {
    const NUM_FEATURES: usize = 162;

    fn active_features(position: &Position, mut activate: impl FnMut(usize)) {
        let stm = position.color;
        let own = position.bitboards[stm as usize] as u32;
        let opp = position.bitboards[(!stm) as usize] as u32;
        let occ = own | opp;

        // 1. SphereOccupancy (0..60)
        for (mut spheres, offset) in [(own, 0), (opp, 30)] {
            while spheres != 0 {
                activate(offset + spheres.trailing_zeros() as usize);
                spheres &= spheres - 1;
            }
        }

        // 2. 2x2 Square configurations (60..102)
        for (s, &sq_mask) in SQUARE_MASKS.iter().enumerate() {
            let base = 60 + s * 3;
            let own_count = (own & sq_mask).count_ones();
            let opp_count = (opp & sq_mask).count_ones();
            if own_count == 3 && opp_count == 0 {
                activate(base);
            } else if opp_count == 3 && own_count == 0 {
                activate(base + 1);
            } else if own_count == 4 {
                activate(base + 2);
            }
        }

        // 3. Placeable locations (102..132) & 4. Unblocked spheres (132..162)
        for i in 0..30 {
            let bit = 1u32 << i;
            if (occ & bit) == 0 {
                if (occ & SUPPORT_MASKS[i]) == SUPPORT_MASKS[i] {
                    activate(102 + i);
                }
            } else if (occ & ABOVE_MASKS[i]) == 0 {
                activate(132 + i);
            }
        }
    }
}