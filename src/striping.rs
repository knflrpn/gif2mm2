use crate::constants::{TARGET_HEIGHT, TARGET_WIDTH};

// Heuristic constraints for striping evaluation.
const COST_V_STRIPE: i32 = 90;
const COST_H_STRIPE: i32 = 160;
const COST_PIXEL: i32 = 4;
const PEN_SIZE: usize = 8;

pub struct Stripe {
    pub is_vertical: bool,
    pub color: u8,
    pub start_cursor: (usize, usize),
    pub end_cursor: (usize, usize),
}

// Determines the benefit or cost of drawing a stripe at a given location
fn eval_v_stripe(x: usize, color: u8, target: &[u8], canvas: &[u8]) -> i32 {
    let mut fixed = 0;
    let mut broken = 0;

    for dx in 0..PEN_SIZE {
        if (x + dx) < 4 {
            break;
        }
        let px = x + dx - 4;
        if px >= TARGET_WIDTH {
            break;
        }

        for y in 0..TARGET_HEIGHT {
            let idx = y * TARGET_WIDTH + px;
            let t = target[idx];
            let c = canvas[idx];

            if c != color {
                if t == color {
                    fixed += 1;
                } else if c == t {
                    broken += 1;
                }
            }
        }
    }
    (fixed * COST_PIXEL) - (broken * COST_PIXEL) - COST_V_STRIPE
}

// Determines the benefit or cost of drawing a stripe at a given location
fn eval_h_stripe(y: usize, color: u8, target: &[u8], canvas: &[u8]) -> i32 {
    let mut fixed = 0;
    let mut broken = 0;

    for dy in 0..PEN_SIZE {
        if (y + dy) < 4 {
            break;
        }
        let py = y + dy - 4;
        if py >= TARGET_HEIGHT {
            break;
        }

        for x in 0..TARGET_WIDTH {
            let idx = py * TARGET_WIDTH + x;
            let t = target[idx];
            let c = canvas[idx];

            if c != color {
                if t == color {
                    fixed += 1;
                } else if c == t {
                    broken += 1;
                }
            }
        }
    }
    (fixed * COST_PIXEL) - (broken * COST_PIXEL) - COST_H_STRIPE
}

fn get_best_v_stripe(x: usize, target: &[u8], canvas: &[u8]) -> (u8, i32) {
    let mut best_color = 0;
    let mut max_benefit = 0;
    for color in 0..crate::constants::PALETTE_SIZE as u8 {
        let benefit = eval_v_stripe(x, color, target, canvas);
        if benefit > max_benefit {
            max_benefit = benefit;
            best_color = color;
        }
    }
    (best_color, max_benefit)
}

fn get_best_h_stripe(y: usize, target: &[u8], canvas: &[u8]) -> (u8, i32) {
    let mut best_color = 0;
    let mut max_benefit = 0;
    for color in 0..crate::constants::PALETTE_SIZE as u8 {
        let benefit = eval_h_stripe(y, color, target, canvas);
        if benefit > max_benefit {
            max_benefit = benefit;
            best_color = color;
        }
    }
    (best_color, max_benefit)
}

// Performs a search for a good set of vertical stripes to draw.  A stripe is
// drawn if it would be at all beneficial and if waiting to draw a stripe
// at the next location would be worse.
pub fn plan_vertical_stripes(
    target: &[u8],
    canvas: &mut [u8],
    start_cursor: (usize, usize),
) -> (Vec<Stripe>, (usize, usize)) {
    let mut stripes = Vec::new();
    let mut cursor_x = start_cursor.0;
    let mut cursor_y = start_cursor.1;

    let mut going_down = cursor_y < (TARGET_HEIGHT / 2);
    let start_at_right = cursor_x >= (TARGET_WIDTH / 2);
    let max_x = TARGET_WIDTH - 1;

    let x_indices: Vec<usize> = if start_at_right {
        (0..=max_x).rev().collect()
    } else {
        (0..=max_x).collect()
    };

    for i in 1..x_indices.len() {
        let x = x_indices[i];
        let (c_curr, benefit_curr) = get_best_v_stripe(x, target, canvas);

        if benefit_curr > 0 {
            let mut should_draw = false;
            let is_last_step = i == x_indices.len() - 1;

            if is_last_step {
                should_draw = true;
            } else {
                let next_x = x_indices[i + 1];
                let benefit_next = eval_v_stripe(next_x, c_curr, target, canvas);

                let abandoned_col = if start_at_right { x + PEN_SIZE - 1 } else { x };

                // Have to consider the fact that moving forward without drawing will permanently
                // abandon the column that goes out of reach.
                let abandoned_benefit = if abandoned_col >= 4 && (abandoned_col - 4) < TARGET_WIDTH
                {
                    eval_v_stripe_col(abandoned_col - 4, c_curr, target, canvas)
                } else {
                    0
                };

                if benefit_next < benefit_curr || abandoned_benefit > 0 {
                    should_draw = true;
                }
            }

            if should_draw {
                let start_y = if going_down { 1 } else { TARGET_HEIGHT - 1 };
                let end_y = if going_down { TARGET_HEIGHT - 1 } else { 1 };

                stripes.push(Stripe {
                    is_vertical: true,
                    color: c_curr,
                    start_cursor: (x, start_y),
                    end_cursor: (x, end_y),
                });

                for dx in 0..PEN_SIZE {
                    if (x + dx) < 4 {
                        break;
                    }
                    let px = x + dx - 4;
                    if px >= TARGET_WIDTH {
                        break;
                    }
                    for y in 0..TARGET_HEIGHT {
                        canvas[y * TARGET_WIDTH + px] = c_curr;
                    }
                }

                cursor_x = x;
                cursor_y = end_y;
                going_down = !going_down;
            }
        }
    }
    (stripes, (cursor_x, cursor_y))
}

// Same as above but for horizontal stripes
pub fn plan_horizontal_stripes(
    target: &[u8],
    canvas: &mut [u8],
    start_cursor: (usize, usize),
) -> (Vec<Stripe>, (usize, usize)) {
    let mut stripes = Vec::new();
    let mut cursor_x = start_cursor.0;
    let mut cursor_y = start_cursor.1;

    let mut going_right = cursor_x < (TARGET_WIDTH / 2);
    let start_at_bottom = cursor_y >= (TARGET_HEIGHT / 2);
    let max_y = TARGET_HEIGHT - 1;

    let y_indices: Vec<usize> = if start_at_bottom {
        (0..=max_y).rev().collect()
    } else {
        (0..=max_y).collect()
    };

    for i in 1..y_indices.len() {
        let y = y_indices[i];
        let (c_curr, benefit_curr) = get_best_h_stripe(y, target, canvas);

        if benefit_curr > 0 {
            let mut should_draw = false;
            let is_last_step = i == y_indices.len() - 1;

            if is_last_step {
                should_draw = true;
            } else {
                let next_y = y_indices[i + 1];
                let benefit_next = eval_h_stripe(next_y, c_curr, target, canvas);

                let abandoned_row = if start_at_bottom { y + PEN_SIZE - 1 } else { y };

                let abandoned_benefit = if abandoned_row >= 4 && (abandoned_row - 4) < TARGET_HEIGHT
                {
                    eval_h_stripe_row(abandoned_row - 4, c_curr, target, canvas)
                } else {
                    0
                };

                if benefit_next < benefit_curr || abandoned_benefit > 0 {
                    should_draw = true;
                }
            }

            if should_draw {
                let start_x = if going_right { 1 } else { TARGET_WIDTH - 1 };
                let end_x = if going_right { TARGET_WIDTH - 1 } else { 1 };

                stripes.push(Stripe {
                    is_vertical: false,
                    color: c_curr,
                    start_cursor: (start_x, y),
                    end_cursor: (end_x, y),
                });

                for dy in 0..PEN_SIZE {
                    if (y + dy) < 4 {
                        break;
                    }
                    let py = y + dy - 4;
                    if py >= TARGET_HEIGHT {
                        break;
                    }
                    for x in 0..TARGET_WIDTH {
                        canvas[py * TARGET_WIDTH + x] = c_curr;
                    }
                }

                cursor_x = end_x;
                cursor_y = y;
                going_right = !going_right;
            }
        }
    }
    (stripes, (cursor_x, cursor_y))
}

fn eval_v_stripe_col(x: usize, color: u8, target: &[u8], canvas: &[u8]) -> i32 {
    let mut fixed = 0;
    let mut broken = 0;
    if x >= TARGET_WIDTH {
        return 0;
    }

    for y in 0..TARGET_HEIGHT {
        let idx = y * TARGET_WIDTH + x;
        let t = target[idx];
        let c = canvas[idx];

        if c != color {
            if t == color {
                fixed += 1;
            } else if c == t {
                broken += 1;
            }
        }
    }
    (fixed * COST_PIXEL) - (broken * COST_PIXEL)
}

fn eval_h_stripe_row(y: usize, color: u8, target: &[u8], canvas: &[u8]) -> i32 {
    let mut fixed = 0;
    let mut broken = 0;
    if y >= TARGET_HEIGHT {
        return 0;
    }

    for x in 0..TARGET_WIDTH {
        let idx = y * TARGET_WIDTH + x;
        let t = target[idx];
        let c = canvas[idx];

        if c != color {
            if t == color {
                fixed += 1;
            } else if c == t {
                broken += 1;
            }
        }
    }
    (fixed * COST_PIXEL) - (broken * COST_PIXEL)
}

pub fn estimate_plan_cost(
    stripes: &[Stripe],
    final_canvas: &[u8],
    target: &[u8],
    cmd_buffer: &mut Vec<u8>,
) -> i32 {
    let mut total_frames = 0;

    if !stripes.is_empty() {
        let mut cursor_x = 0;
        let mut cursor_y = 0;
        let mut current_color = 0;

        total_frames += 24;

        for stripe in stripes {
            let tx = stripe.start_cursor.0;
            let ty = stripe.start_cursor.1;

            let mut stick_frames = 0;
            let mut new_cx = cursor_x;
            let mut new_cy = cursor_y;

            let max_x = (TARGET_WIDTH - 1) as i32;
            let max_y = (TARGET_HEIGHT - 1) as i32;

            let cost_x_normal = (cursor_x as i32 - tx as i32).abs() * 2;
            let frames_left = (cursor_x as i32 * COST_H_STRIPE) / TARGET_WIDTH as i32;
            let cost_x_left = frames_left + (tx as i32 * 2);
            let frames_right = ((max_x - cursor_x as i32) * COST_H_STRIPE) / TARGET_WIDTH as i32;
            let cost_x_right = frames_right + ((max_x - tx as i32) * 2);

            if cost_x_left < cost_x_normal && cost_x_left <= cost_x_right && frames_left > 0 {
                stick_frames += frames_left;
                new_cx = 1;
            } else if cost_x_right < cost_x_normal && frames_right > 0 {
                stick_frames += frames_right;
                new_cx = max_x as usize;
            }

            let cost_y_normal = (cursor_y as i32 - ty as i32).abs() * 2;
            let frames_up = (cursor_y as i32 * COST_V_STRIPE) / TARGET_HEIGHT as i32;
            let cost_y_up = frames_up + (ty as i32 * 2);
            let frames_down = ((max_y - cursor_y as i32) * COST_V_STRIPE) / TARGET_HEIGHT as i32;
            let cost_y_down = frames_down + ((max_y - ty as i32) * 2);

            if cost_y_up < cost_y_normal && cost_y_up <= cost_y_down && frames_up > 0 {
                stick_frames += frames_up;
                new_cy = 1;
            } else if cost_y_down < cost_y_normal && frames_down > 0 {
                stick_frames += frames_down;
                new_cy = max_y as usize;
            }

            let dx = tx as i16 - new_cx as i16;
            let dy = ty as i16 - new_cy as i16;
            let c_delta = crate::color::get_color_dist(current_color, stripe.color);

            crate::commands::get_move_commands(dx, dy, 0, cmd_buffer);
            crate::commands::get_color_change_commands(c_delta, 0, cmd_buffer);

            total_frames += stick_frames;
            total_frames += (cmd_buffer.len() as i32) * 2;
            total_frames += if stripe.is_vertical {
                COST_V_STRIPE
            } else {
                COST_H_STRIPE
            };

            cursor_x = stripe.end_cursor.0;
            cursor_y = stripe.end_cursor.1;
            current_color = stripe.color;
        }
    }

    let mut incorrect_pixels = 0;
    for i in 0..final_canvas.len() {
        if final_canvas[i] != target[i] {
            incorrect_pixels += 1;
        }
    }
    total_frames += incorrect_pixels * COST_PIXEL;
    total_frames
}
