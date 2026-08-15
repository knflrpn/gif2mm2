use crate::CostCache;
use crate::commands::derive_commands;
use crate::constants::{TARGET_HEIGHT, TARGET_WIDTH};
use crate::color::get_color_dist;

pub const INITIAL_N: usize = 7;

#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub weight_x: f32, // How much to prefer moving horizontally
    pub weight_y: f32, // and vertically
    pub bias_x: f32,   // How much to favor right (+) or left (-)
    pub bias_y: f32,   // or up or down
}

// Breaks frame ties based on biases
fn calculate_tie_score(dx: i16, dy: i16, params: &SearchParams) -> f32 {
    let abs_x = dx.abs() as f32;
    let abs_y = dy.abs() as f32;
    (params.weight_x * abs_x) + (params.weight_y * abs_y)
        - (params.bias_x * dx as f32)
        - (params.bias_y * dy as f32)
}

// Pushes searches in directions based on biases
fn calculate_weighted_distance(dx: i16, dy: i16, params: &SearchParams) -> f32 {
    let wx = params.weight_x * dx as f32;
    let wy = params.weight_y * dy as f32;
    (wx * wx + wy * wy).sqrt() - (params.bias_x * dx as f32) - (params.bias_y * dy as f32)
}

// Perform a search in increasing radii to find a good pixel to go to next
fn find_next_pixel(
    canvas: &[u8],
    target: &[u8],
    cx: usize,
    cy: usize,
    current_color: u8,
    prev_cmd: u8,
    params: &SearchParams,
    cache: &mut CostCache,
    cmd_buffer: &mut Vec<u8>,
) -> Option<(usize, usize, u8, usize)> {
    assert!(canvas.len() >= TARGET_WIDTH * TARGET_HEIGHT);
    assert!(target.len() >= TARGET_WIDTH * TARGET_HEIGHT);
    assert!(cx < TARGET_WIDTH && cy < TARGET_HEIGHT);

    // Radii start and N and double twice before giving up
    let radii = [INITIAL_N, INITIAL_N * 2, INITIAL_N * 4];

    for &radius in &radii {
        let min_x = cx.saturating_sub(radius);
        let max_x = (cx + radius).min(TARGET_WIDTH - 1);
        let min_y = cy.saturating_sub(radius);
        let max_y = (cy + radius).min(TARGET_HEIGHT - 1);

        let mut best_score = f32::MAX;
        let mut best_coord = None;

        for y in min_y..(max_y + 1) {
            let row_idx = y * TARGET_WIDTH;

            for x in min_x..(max_x + 1) {
                let idx = row_idx + x;

                if canvas[idx] != target[idx] {
                    let target_color = target[idx];
                    let dx = x as i16 - cx as i16;
                    let dy = y as i16 - cy as i16;
                    let color_delta = get_color_dist(current_color, target_color);

                    let (cost, _) = cache.get_or_insert_with(
                        dx,
                        dy,
                        color_delta,
                        prev_cmd,
                        cmd_buffer,
                        |buf| {
                            derive_commands(dx, dy, color_delta, prev_cmd, buf);
                        },
                    );

                    let directional_modifier = calculate_tie_score(dx, dy, params);
                    let score = (cost as f32) + directional_modifier;

                    if score < best_score {
                        best_score = score;
                        best_coord = Some((x, y));
                    }
                }
            }
        }

        if let Some((best_x, best_y)) = best_coord {
            let idx = best_y * TARGET_WIDTH + best_x;
            let target_color = target[idx];
            let dx = best_x as i16 - cx as i16;
            let dy = best_y as i16 - cy as i16;
            let color_delta = get_color_dist(current_color, target_color);

            derive_commands(dx, dy, color_delta, prev_cmd, cmd_buffer);
            let last_cmd = cmd_buffer.last().copied().unwrap_or(0);
            return Some((best_x, best_y, last_cmd, cmd_buffer.len()));
        }
    }


    // Nothing found -- pick the closest in the whole image.
    let mut best_pixel: Option<(usize, usize, f32)> = None;
    let canvas_chunks = canvas.chunks_exact(TARGET_WIDTH);
    let target_chunks = target.chunks_exact(TARGET_WIDTH);

    for (y, (c_row, t_row)) in canvas_chunks.zip(target_chunks).enumerate() {
        for (x, (&c, &t)) in c_row.iter().zip(t_row.iter()).enumerate() {
            if c != t {
                let dx = x as i16 - cx as i16;
                let dy = y as i16 - cy as i16;
                let dist = calculate_weighted_distance(dx, dy, params);

                if best_pixel.map_or(true, |(_, _, min_dist)| dist < min_dist) {
                    best_pixel = Some((x, y, dist));
                }
            }
        }
    }

    if let Some((x, y, _)) = best_pixel {
        let idx = y * TARGET_WIDTH + x;
        let target_color = target[idx];
        let dx = x as i16 - cx as i16;
        let dy = y as i16 - cy as i16;
        let color_delta = get_color_dist(current_color, target_color);

        derive_commands(dx, dy, color_delta, prev_cmd, cmd_buffer);
        let last_cmd = cmd_buffer.last().copied().unwrap_or(0);
        return Some((x, y, last_cmd, cmd_buffer.len()));
    }

    None
}

// Searches for pixel after pixel until whole image is complete
pub fn plan_pixel_path(
    target: &[u8],
    canvas: &mut [u8],
    mut cx: usize,
    mut cy: usize,
    mut current_color: u8,
    params: &SearchParams,
    cache: &mut CostCache,
    cmd_buffer: &mut Vec<u8>,
) -> (Vec<(usize, usize)>, usize) {
    let mut path = Vec::new();
    let mut prev_cmd = 0u8;
    let mut total_cost = 0;

    while let Some((px, py, next_prev_cmd, step_cost)) = find_next_pixel(
        canvas,
        target,
        cx,
        cy,
        current_color,
        prev_cmd,
        params,
        cache,
        cmd_buffer,
    ) {
        let idx = py * TARGET_WIDTH + px;
        canvas[idx] = target[idx];

        cx = px;
        cy = py;
        current_color = target[idx];
        prev_cmd = next_prev_cmd;
        total_cost += step_cost;

        path.push((px, py));
    }

    (path, total_cost)
}

// Given a sequence of pixels, generates the controller states needed to achieve it.
pub fn generate_commands_from_path(
    path: &[(usize, usize)],
    target: &[u8],
    mut cx: usize,
    mut cy: usize,
    mut current_color: u8,
    cmd_buffer: &mut Vec<u8>,
) -> Vec<u8> {
    let mut all_commands = Vec::new();
    let mut prev_cmd = 0u8;

    for &(px, py) in path {
        let idx = py * TARGET_WIDTH + px;
        let target_color = target[idx];
        let dx = px as i16 - cx as i16;
        let dy = py as i16 - cy as i16;
        let color_delta = get_color_dist(current_color, target_color);

        derive_commands(dx, dy, color_delta, prev_cmd, cmd_buffer);
        if let Some(&last_cmd) = cmd_buffer.last() {
            prev_cmd = last_cmd;
        }
        all_commands.extend_from_slice(cmd_buffer);

        cx = px;
        cy = py;
        current_color = target_color;
    }

    all_commands
}
