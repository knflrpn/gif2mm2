// src/constants.rs

// Global constraints and dimensions.
pub const TARGET_WIDTH: usize = 320;
pub const TARGET_HEIGHT: usize = 180;
pub const PALETTE_SIZE: i8 = 17;

// Bitmasks for controller buttons.
pub const C_A: u8 = 1 << 0;
pub const C_L: u8 = 1 << 1;
pub const C_R: u8 = 1 << 2;
pub const C_U: u8 = 1 << 3;
pub const C_D: u8 = 1 << 4;
pub const C_L2: u8 = 1 << 5;
pub const C_R2: u8 = 1 << 6;