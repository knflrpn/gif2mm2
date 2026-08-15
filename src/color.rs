use crate::constants::PALETTE_SIZE;

// The Mario Maker 2 palette of colors.
pub const PALETTE: &[(u8, u8, u8)] = &[
    (0xfe, 0x00, 0x00),
    (0xbc, 0x01, 0x01),
    (0xff, 0xf5, 0xd3),
    (0xad, 0x80, 0x47),
    (0xfe, 0xff, 0x00),
    (0xfd, 0xc2, 0x00),
    (0x09, 0xff, 0x00),
    (0x00, 0xbc, 0x05),
    (0x00, 0xff, 0xff),
    (0x0a, 0x00, 0xfe),
    (0xbb, 0x62, 0xff),
    (0x8a, 0x00, 0xbc),
    (0xfe, 0xc2, 0xfe),
    (0xba, 0x07, 0x92),
    (0xbb, 0xbc, 0xba),
    (0x00, 0x00, 0x00),
    (0xff, 0xff, 0xff),
];

// Find the index of the closest color in the palette to the given RGB color.
pub fn find_closest_color(color: (u8, u8, u8), palette: &[(u8, u8, u8)]) -> u8 {
    let mut closest_index = 0;
    let mut closest_distance = u32::MAX;
    for (index, &palette_color) in palette.iter().enumerate() {
        let distance = color_distance_squared(color, palette_color);
        if distance < closest_distance {
            closest_distance = distance;
            closest_index = index;
        }
    }
    closest_index as u8
}

// Estimate the visual similarity between two colors using squared Euclidean distance.
pub fn color_distance_squared(c1: (u8, u8, u8), c2: (u8, u8, u8)) -> u32 {
    let r_diff = c1.0 as i32 - c2.0 as i32;
    let g_diff = c1.1 as i32 - c2.1 as i32;
    let b_diff = c1.2 as i32 - c2.2 as i32;
    (r_diff * r_diff + g_diff * g_diff + b_diff * b_diff) as u32
}

// Find the minimum number of steps to change from the current color to the desired color,
// considering the circular nature of the palette.
pub fn get_color_dist(current: u8, desired: u8) -> i8 {
    let diff = desired as i8 - current as i8;
    let half_palette = PALETTE_SIZE / 2;

    if diff < -half_palette {
        diff + PALETTE_SIZE
    } else if diff > half_palette {
        diff - PALETTE_SIZE
    } else {
        diff
    }
}