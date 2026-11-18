use std::ops::Not;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    #[default]
    White = 0,
    Black = 1,
}

impl Not for Color {
    type Output = Self;

    fn not(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum GamePhase {
    #[default]
    Move = 0,
    RemoveFirst = 1,
    RemoveSecond = 2,
}

/// Thirty packed sphere locations per color, ordered white then black.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Position {
    pub bitboards: [u32; 2],
    pub phase: GamePhase,
    pub color: Color,
}

impl From<u64> for Position {
    fn from(packed: u64) -> Self {
        const BOARD_MASK: u64 = (1 << 30) - 1;
        Self {
            bitboards: [
                (packed & BOARD_MASK) as u32,
                ((packed >> 30) & BOARD_MASK) as u32,
            ],
            phase: match packed >> 61 {
                0 => GamePhase::Move,
                1 => GamePhase::RemoveFirst,
                2 => GamePhase::RemoveSecond,
                _ => panic!("invalid packed phase or reserved position bit"),
            },
            color: if packed & (1 << 60) == 0 {
                Color::White
            } else {
                Color::Black
            },
        }
    }
}

impl From<Position> for u64 {
    fn from(position: Position) -> Self {
        u64::from(position.bitboards[0])
            | (u64::from(position.bitboards[1]) << 30)
            | ((position.color as u64) << 60)
            | ((position.phase as u64) << 61)
    }
}
