// src/main.rs
extern crate image;
use ahash::AHashMap;
use image::GenericImageView;
use std::env;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::rc::Rc;

mod utils {
    pub fn calculate_coord_cost(sequence_length: u16, x: i16, y: i16, length_weight: f32, x_weight: f32, y_weight: f32) -> u16 {
        (sequence_length as f32 * length_weight + x as f32 * x_weight + y as f32 * y_weight + 10000.0) as u16
    }

    pub fn get_color_dist(current: u8, desired: u8) -> i8 {
        let mut move_dist = desired as i8 - current as i8;
        if move_dist < -8 {
            move_dist += 17;
        }
        if move_dist > 8 {
            move_dist -= 17;
        }
        move_dist
    }
}

mod commands {
    use super::{C_A, C_D, C_L, C_L2, C_R, C_R2, C_U};

    pub fn get_color_change_commands(color_delta: i8, prev_cmd: u8) -> Vec<u8> {
        let mut moves = Vec::with_capacity(16);
        if color_delta == 0 {
            return moves;
        }
        let mut add_blank = prev_cmd & (C_R2 | C_L2) != 0;
        let mut need = color_delta;
        while need != 0 {
            if add_blank {
                moves.push(0);
            }
            add_blank = true;
            if need > 0 {
                moves.push(C_R2);
                need -= 1;
            } else {
                moves.push(C_L2);
                need += 1;
            }
        }
        moves
    }

    pub fn cmd_to_string(cmd: u8) -> String {
        let mut s = String::new();
        if (cmd & C_A) != 0 {
            s.push_str("A ");
        }
        if (cmd & C_L) != 0 {
            s.push_str("L ");
        }
        if (cmd & C_R) != 0 {
            s.push_str("R ");
        }
        if (cmd & C_U) != 0 {
            s.push_str("U ");
        }
        if (cmd & C_D) != 0 {
            s.push_str("D ");
        }
        if (cmd & C_L2) != 0 {
            s.push_str("L2 ");
        }
        if (cmd & C_R2) != 0 {
            s.push_str("R2 ");
        }
        s
    }

    pub fn elementwise_or(a: &[u8], b: &[u8]) -> Vec<u8> {
        let capacity = std::cmp::max(a.len(), b.len());
        let mut result = Vec::with_capacity(capacity);
        for i in 0..capacity {
            let a_value = if i < a.len() { a[i] } else { 0 };
            let b_value = if i < b.len() { b[i] } else { 0 };
            result.push(a_value | b_value);
        }
        result
    }

    pub fn get_move_commands(mut d_x: i16, mut d_y: i16, prev_cmd: u8) -> Vec<u8> {
        let mut commands = Vec::with_capacity(16);
        let mut last_cmd = prev_cmd;
        while d_x != 0 || d_y != 0 {
            let (mut potential_cmd_x, mut potential_cmd_y) = (0, 0);
            let mut potential_x = 0;
            let mut potential_y = 0;
            let mut progress = false;
            if (d_x > 0) && ((C_R & last_cmd) == 0) {
                potential_cmd_x = C_R;
                potential_x = 1;
            } else if (d_x < 0) && ((C_L & last_cmd) == 0) {
                potential_cmd_x = C_L;
                potential_x = -1;
            }
            if (d_y < 0) && ((C_U & last_cmd) == 0) {
                potential_cmd_y = C_U;
                potential_y = -1;
            } else if (d_y > 0) && ((C_D & last_cmd) == 0) {
                potential_cmd_y = C_D;
                potential_y = 1;
            }
            if d_y.abs() > d_x.abs() {
                if potential_cmd_y != 0 {
                    commands.push(potential_cmd_y);
                    d_y -= potential_y;
                    last_cmd = potential_cmd_y;
                    progress = true;
                }
                if potential_cmd_x != 0 {
                    commands.push(potential_cmd_x);
                    d_x -= potential_x;
                    last_cmd = potential_cmd_x;
                    progress = true;
                }
            } else {
                if potential_cmd_x != 0 {
                    commands.push(potential_cmd_x);
                    d_x -= potential_x;
                    last_cmd = potential_cmd_x;
                    progress = true;
                }
                if potential_cmd_y != 0 {
                    commands.push(potential_cmd_y);
                    d_y -= potential_y;
                    last_cmd = potential_cmd_y;
                    progress = true;
                }
            }
            if !progress {
                commands.push(0);
                last_cmd = 0;
            }
        }
        commands
    }

    pub fn derive_commands(
        pixel_offset_x: i16,
        pixel_offset_y: i16,
        color_delta: i8,
        prev_cmd: u8,
    ) -> Vec<u8> {
        let move_commands = get_move_commands(pixel_offset_x, pixel_offset_y, prev_cmd);
        let color_commands = get_color_change_commands(color_delta, prev_cmd);
        let mut full_commands = elementwise_or(&move_commands, &color_commands);
        if (full_commands.len() == 1) && (prev_cmd & C_A != 0) {
            if full_commands[0] & (C_L2 | C_R2) != 0 {
                full_commands.push(0);
            }
        }
        if full_commands.is_empty() {
            full_commands.push(C_A);
        } else {
            let final_index = full_commands.len() - 1;
            full_commands[final_index] += C_A;
        }
        full_commands
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct CacheKey {
    pub pixel_offset_x: i16,
    pub pixel_offset_y: i16,
    pub color_delta: i8,
    pub prev_cmd: u8,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct CacheValue {
    pub result: Rc<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct CommandCache {
    pub cache: AHashMap<CacheKey, CacheValue>,
    pub cost_cache: AHashMap<CacheKey, (u16, u8)>,
}

impl CommandCache {
    pub fn new() -> Self {
        CommandCache {
            cache: AHashMap::new(),
            cost_cache: AHashMap::new(),
        }
    }

    /// Computes the command sequence and caches both the sequence (as an Rc) and its cost.
    fn compute_and_cache(
        &mut self,
        key: CacheKey,
        pixel_offset_x: i16,
        pixel_offset_y: i16,
        color_delta: i8,
        prev_cmd: u8,
    ) -> (Rc<Vec<u8>>, (u16, u8)) {
        let result =
            commands::derive_commands(pixel_offset_x, pixel_offset_y, color_delta, prev_cmd);
        let rc_result = Rc::new(result);
        let cost = (rc_result.len() as u16, *rc_result.last().unwrap_or(&0));
        self.cache.insert(
            key.clone(),
            CacheValue {
                result: rc_result.clone(),
            },
        );
        self.cost_cache.insert(key, cost);
        (rc_result, cost)
    }

    /// Returns a shared pointer to the command sequence for the given parameters.
    pub fn get_result(
        &mut self,
        pixel_offset_x: i16,
        pixel_offset_y: i16,
        color_delta: i8,
        prev_cmd: u8,
    ) -> Rc<Vec<u8>> {
        let key = CacheKey {
            pixel_offset_x,
            pixel_offset_y,
            color_delta,
            prev_cmd,
        };
        if std::cmp::max(pixel_offset_x.abs(), pixel_offset_y.abs()) < 16 {
            if let Some(value) = self.cache.get(&key) {
                return value.result.clone();
            }
            let (rc_result, _cost) =
                self.compute_and_cache(key, pixel_offset_x, pixel_offset_y, color_delta, prev_cmd);
            return rc_result;
        }
        // If the pixel offset is large, compute without caching.
        Rc::new(commands::derive_commands(
            pixel_offset_x,
            pixel_offset_y,
            color_delta,
            prev_cmd,
        ))
    }

    /// Returns the cost (length and last command) for the command sequence.
    pub fn get_cost(
        &mut self,
        pixel_offset_x: i16,
        pixel_offset_y: i16,
        color_delta: i8,
        prev_cmd: u8,
    ) -> (u16, u8) {
        let key = CacheKey {
            pixel_offset_x,
            pixel_offset_y,
            color_delta,
            prev_cmd,
        };
        if std::cmp::max(pixel_offset_x.abs(), pixel_offset_y.abs()) < 16 {
            if let Some(cost) = self.cost_cache.get(&key) {
                return *cost;
            }
            let (_rc_result, cost) =
                self.compute_and_cache(key, pixel_offset_x, pixel_offset_y, color_delta, prev_cmd);
            return cost;
        }
        let sequence =
            commands::derive_commands(pixel_offset_x, pixel_offset_y, color_delta, prev_cmd);
        (sequence.len() as u16, sequence.last().copied().unwrap_or(0))
    }
}

mod pathfinding {
    use super::utils;
    use super::CommandCache;
    use ahash::AHashMap;

    pub fn closest_undrawn_points(
        x: i16,
        y: i16,
        max_dist: i16,
        max_count: i16,
        drawn_pixels: &AHashMap<(i16, i16), bool>,
    ) -> Vec<(i16, i16)> {
        let max_dist = if max_dist > (320+180) { 320 } else { max_dist };
        let mut points = Vec::new();
        let mut got_count = 0;
        for d in 1..max_dist {
            for i in 0..d {
                let test_points = [
                    (x + d - i, y + i),
                    (x - i, y + d - i),
                    (x - d + i, y - i),
                    (x + i, y - d + i),
                ];
                for &(nx, ny) in &test_points {
                    if !drawn_pixels.get(&(nx, ny)).unwrap_or(&true) {
                        points.push((nx, ny));
                        got_count += 1;
                    }
                }
            }
            if got_count >= max_count {
                break;
            }
        }
        points
    }

    pub fn greedy_coord_search(
        current_x: i16,
        current_y: i16,
        current_color: u8,
        prev_cmd: u8,
        frame_buffer: &[[u8; 180]; 320],
        drawn_pixels: &AHashMap<(i16, i16), bool>,
        cache: &mut CommandCache,
        length_weight: f32,
        x_weight: f32,
        y_weight: f32,
    ) -> Option<(i16, i16)> {
        let potential_coords = {
            let mut pts = closest_undrawn_points(current_x, current_y, 5, 100, drawn_pixels);
            if pts.is_empty() {
                pts = closest_undrawn_points(current_x, current_y, 16, 100, drawn_pixels);
            }
            if pts.is_empty() {
                pts = closest_undrawn_points(current_x, current_y, 50, 200, drawn_pixels);
            }
            if pts.is_empty() {
                pts = closest_undrawn_points(current_x, current_y, 320+180, 200, drawn_pixels);
            }
            pts
        };
        if potential_coords.is_empty() {
            return None;
        }
        let mut best_point = potential_coords[0];
        let color_delta = utils::get_color_dist(
            current_color,
            frame_buffer[best_point.0 as usize][best_point.1 as usize],
        );
        let mut sequence_length = cache
            .get_cost(
                best_point.0 - current_x,
                best_point.1 - current_y,
                color_delta,
                prev_cmd,
            )
            .0;
        let mut min_cost =
            utils::calculate_coord_cost(sequence_length, best_point.0, best_point.1, length_weight, x_weight, y_weight);
        for &(x, y) in &potential_coords {
            let color_delta =
                utils::get_color_dist(current_color, frame_buffer[x as usize][y as usize]);
            sequence_length = cache
                .get_cost(x - current_x, y - current_y, color_delta, prev_cmd)
                .0;
            let cost = utils::calculate_coord_cost(sequence_length, x, y, length_weight, x_weight, y_weight);
            if cost < min_cost {
                best_point = (x, y);
                min_cost = cost;
            }
        }
        Some(best_point)
    }
}

mod optimizer {
    use super::CommandCache;
    use crate::C_A;
    use std::cmp;

    // Find small isolated blocks of pixels within the overall sequence and re–insert those blocks at better locations.
    pub fn rearrange_pixels(
        visit_order: &mut Vec<(i16, i16)>,
        isolated_block_size: usize,
        isolation_distance: i16,
    ) {
        let mut isolated_blocks: Vec<Vec<(i16, i16)>> = Vec::new();
        let mut i = 0;
        while i < visit_order.len() {
            let start_index = i;
            let mut end_index = i;
            let (mut prev_x, mut prev_y) = visit_order[i];
            while end_index < visit_order.len() - 1 {
                let (next_x, next_y) = visit_order[end_index + 1];
                let x_dist = (next_x - prev_x).abs();
                let y_dist = (next_y - prev_y).abs();
                if x_dist > isolation_distance || y_dist > isolation_distance {
                    break;
                }
                prev_x = next_x;
                prev_y = next_y;
                end_index += 1;
            }
            let block_size = end_index - start_index + 1;
            if block_size <= isolated_block_size {
                let block = visit_order.drain(start_index..=end_index).collect();
                isolated_blocks.push(block);
                i = start_index;
            } else {
                i = end_index + 1;
            }
        }
        for block in isolated_blocks {
            let nearest_index = find_nearest_index(&visit_order, &block[0]);
            visit_order.splice(nearest_index..nearest_index, block);
        }
    }

    pub fn find_nearest_index(visit_order: &Vec<(i16, i16)>, coord: &(i16, i16)) -> usize {
        let mut nearest_index = 0;
        let mut min_distance = i16::MAX;
        for (j, &(visit_x, visit_y)) in visit_order.iter().enumerate() {
            let distance = cmp::max((coord.0 - visit_x).abs(), (coord.1 - visit_y).abs());
            if j < visit_order.len() - 1 {
                let next_coord = visit_order[j + 1];
                let next_distance = cmp::max(
                    (next_coord.0 - visit_x).abs(),
                    (next_coord.1 - visit_y).abs(),
                );
                if distance < min_distance && next_distance >= 2 {
                    min_distance = distance;
                    nearest_index = j;
                }
            }
        }
        nearest_index
    }

    pub fn generate_and_search_permutations(
        current_permutation: &mut Vec<(i16, i16)>,
        remaining_coords: &mut Vec<(i16, i16)>,
        current_cost: u16,
        optimal_cost: &mut u16,
        optimal_permutation: &mut Option<Vec<(i16, i16)>>,
        start_coord: (i16, i16),
        end_coord: (i16, i16),
        frame_buffer: &[[u8; 180]; 320],
        cache: &mut CommandCache,
        prev_cmd: u8,
    ) {
        let lower_bound = (remaining_coords.len() as f32 * 1.3) as u16;
        if current_cost + lower_bound >= *optimal_cost {
            return;
        }
        if remaining_coords.is_empty() {
            let last_coord = *current_permutation.last().unwrap();
            let temp_color = frame_buffer[last_coord.0 as usize][last_coord.1 as usize];
            let last_color = frame_buffer[end_coord.0 as usize][end_coord.1 as usize];
            let last_commands = cache.get_cost(
                end_coord.0 - last_coord.0,
                end_coord.1 - last_coord.1,
                super::utils::get_color_dist(temp_color, last_color),
                prev_cmd,
            );
            let cost = current_cost + last_commands.0;
            if cost < *optimal_cost {
                *optimal_cost = cost;
                *optimal_permutation = Some(current_permutation.clone());
            }
            return;
        }

        let n = remaining_coords.len();
        for i in 0..n {
            remaining_coords.swap(i, n - 1);
            let coord = remaining_coords.pop().unwrap();
            let last_coord = if current_permutation.is_empty() {
                start_coord
            } else {
                *current_permutation.last().unwrap()
            };
            let temp_color = frame_buffer[last_coord.0 as usize][last_coord.1 as usize];
            let next_color = frame_buffer[coord.0 as usize][coord.1 as usize];
            let command_cost = cache.get_cost(
                coord.0 - last_coord.0,
                coord.1 - last_coord.1,
                super::utils::get_color_dist(temp_color, next_color),
                prev_cmd,
            );
            let new_cost = current_cost + command_cost.0;
            let new_prev_cmd = command_cost.1;
            current_permutation.push(coord);

            generate_and_search_permutations(
                current_permutation,
                remaining_coords,
                new_cost,
                optimal_cost,
                optimal_permutation,
                start_coord,
                end_coord,
                frame_buffer,
                cache,
                new_prev_cmd,
            );

            current_permutation.pop();
            remaining_coords.push(coord);
            remaining_coords.swap(i, n - 1);
        }
    }

    pub fn optimize_segment(
        visit_order: &mut Vec<(i16, i16)>,
        index: usize,
        length: usize,
        frame_buffer: &[[u8; 180]; 320],
        cache: &mut CommandCache,
    ) {
        if length <= 3 {
            return;
        }
        let start_coord = visit_order[index];
        let end_coord = visit_order[index + length - 1];
        let mut middle_coords: Vec<(i16, i16)> =
            visit_order[index + 1..index + length - 1].to_vec();
        let prev_prev_cmd = if index < 1 {
            0
        } else {
            let prev_coord = visit_order[index];
            let prev_prev_coord = visit_order[index - 1];
            let prev_color = frame_buffer[prev_coord.0 as usize][prev_coord.1 as usize];
            let prev_prev_color =
                frame_buffer[prev_prev_coord.0 as usize][prev_prev_coord.1 as usize];
            let commands = cache.get_cost(
                prev_coord.0 - prev_prev_coord.0,
                prev_coord.1 - prev_prev_coord.1,
                super::utils::get_color_dist(prev_prev_color, prev_color),
                C_A,
            );
            commands.1
        };
        let mut optimal_permutation: Option<Vec<(i16, i16)>> = None;
        let mut optimal_cost = u16::MAX;
        let mut current_permutation = Vec::with_capacity(middle_coords.len());
        generate_and_search_permutations(
            &mut current_permutation,
            &mut middle_coords,
            0,
            &mut optimal_cost,
            &mut optimal_permutation,
            start_coord,
            end_coord,
            frame_buffer,
            cache,
            prev_prev_cmd,
        );
        if let Some(middle_permutation) = optimal_permutation {
            let new_order = [start_coord]
                .iter()
                .cloned()
                .chain(middle_permutation)
                .chain(std::iter::once(end_coord))
                .collect::<Vec<_>>();
            visit_order.splice(index..index + length, new_order);
        }
    }

    pub fn optimize_chunks(visit_order: &mut Vec<(i16, i16)>, gap_distance: i16) {
        let mut chunks: Vec<Vec<(i16, i16)>> = Vec::new();
        let mut current_chunk: Vec<(i16, i16)> = Vec::new();
        for i in 0..visit_order.len() {
            current_chunk.push(visit_order[i]);
            if i < visit_order.len() - 1 {
                let curr_pixel = visit_order[i];
                let next_pixel = visit_order[i + 1];
                let distance = std::cmp::max(
                    (curr_pixel.0 - next_pixel.0).abs(),
                    (curr_pixel.1 - next_pixel.1).abs(),
                );
                if distance > gap_distance {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                }
            }
        }
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }
        let mut new_visit_order: Vec<(i16, i16)> = chunks.remove(0);
        while !chunks.is_empty() {
            let end_coord = new_visit_order.last().unwrap().clone();
            let mut min_distance = i16::MAX;
            let mut min_index = 0;
            for (i, chunk) in chunks.iter().enumerate() {
                let start_coord = chunk[0];
                let distance = std::cmp::max(
                    (start_coord.0 - end_coord.0).abs(),
                    (start_coord.1 - end_coord.1).abs(),
                );
                if distance < min_distance {
                    min_distance = distance;
                    min_index = i;
                }
            }
            let next_chunk = chunks.remove(min_index);
            new_visit_order.extend(next_chunk);
        }
        *visit_order = new_visit_order;
    }
}

mod color {
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

    pub fn color_distance_squared(c1: (u8, u8, u8), c2: (u8, u8, u8)) -> u32 {
        let r_diff = c1.0 as i32 - c2.0 as i32;
        let g_diff = c1.1 as i32 - c2.1 as i32;
        let b_diff = c1.2 as i32 - c2.2 as i32;
        (r_diff * r_diff + g_diff * g_diff + b_diff * b_diff) as u32
    }

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
}

const C_A: u8 = 1 << 0;
const C_L: u8 = 1 << 1;
const C_R: u8 = 1 << 2;
const C_U: u8 = 1 << 3;
const C_D: u8 = 1 << 4;
const C_L2: u8 = 1 << 5;
const C_R2: u8 = 1 << 6;

//
// Helper structs and functions for refactoring main
//

/// Parse command‐line arguments.
fn parse_args() -> (String, u32, u32) {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!(
            "Usage: {} <input_file> <frames_per_command> (optional, default 2)",
            args[0]
        );
        std::process::exit(1);
    }
    let input_file = args[1].clone();
    let frames_per_command = if args.len() >= 3 {
        args[2].parse().unwrap_or(2)
    } else {
        2
    };
    let optimize = if args.len() >= 4 {
        args[3].parse().unwrap_or(12)
    } else {
        6
    };
    (input_file, frames_per_command, optimize)
}

/// Load the image (or exit on error)
fn load_image(input_file: &str) -> image::DynamicImage {
    match image::open(input_file) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Error opening file: {}", e);
            std::process::exit(1);
        }
    }
}

/// Create the 320×180 frame buffer (with default fill color 16) by cropping and
/// mapping the image colors via the palette.
fn create_frame_buffer(img: &image::DynamicImage) -> [[u8; 180]; 320] {
    let (width, height) = img.dimensions();
    const TARGET_WIDTH: u32 = 320;
    const TARGET_HEIGHT: u32 = 180;
    const FILL_COLOR_INDEX: u8 = 16;
    let mut frame_buffer = [[FILL_COLOR_INDEX; 180]; 320];

    let src_offset_x = if width > TARGET_WIDTH {
        (width - TARGET_WIDTH) / 2
    } else {
        0
    };
    let src_offset_y = if height > TARGET_HEIGHT {
        (height - TARGET_HEIGHT) / 2
    } else {
        0
    };
    let fb_offset_x = if width < TARGET_WIDTH {
        (TARGET_WIDTH - width) / 2
    } else {
        0
    };
    let fb_offset_y = if height < TARGET_HEIGHT {
        (TARGET_HEIGHT - height) / 2
    } else {
        0
    };

    for x in fb_offset_x..(TARGET_WIDTH - fb_offset_x) {
        for y in fb_offset_y..(TARGET_HEIGHT - fb_offset_y) {
            let src_x = x + src_offset_x - fb_offset_x;
            let src_y = y + src_offset_y - fb_offset_y;
            if src_x < width && src_y < height {
                let rgba = img.get_pixel(src_x, src_y);
                let (r, g, b, _) = (rgba[0], rgba[1], rgba[2], rgba[3]);
                let closest_color_index = color::find_closest_color((r, g, b), color::PALETTE);
                frame_buffer[x as usize][y as usize] = closest_color_index;
            } else {
                frame_buffer[x as usize][y as usize] = FILL_COLOR_INDEX;
            }
        }
    }
    frame_buffer
}

/// Structure to hold the results of the stripe–filling phase.
struct StripeResult {
    output: String,
    drawn_pixels: AHashMap<(i16, i16), bool>,
    num_to_draw: u32,
    current_x: i16,
    current_y: i16,
    current_color: u8,
    prev_cmd: u8,
    total_frames: u32,
}

/// Apply the vertical/horizontal stripe–filling strategy (choosing the one that “fills” more pixels).
fn apply_stripes(frame_buffer: &[[u8; 180]; 320], frames_per_command: u32) -> StripeResult {
    // --- Compute vertical stripe candidates ---
    let mut vertical_bg_colors: [u8; 40] = [16; 40];
    for x in 0..40 {
        let mut color_counts: AHashMap<u8, usize> = AHashMap::new();
        color_counts.insert(16, 0);
        for x_inner in (x * 8)..((x + 1) * 8) {
            for y in 0..180 {
                let color_val = frame_buffer[x_inner][y];
                *color_counts.entry(color_val).or_insert(0) += 1;
            }
        }
        let white_count = *color_counts.get(&16).unwrap();
        let (mut max_color, mut max_count) = (16, 0);
        for (col, count) in &color_counts {
            if *count > max_count {
                max_color = *col;
                max_count = *count;
            }
        }
        if max_color != 16 && max_count > (white_count + white_count / 16) {
            vertical_bg_colors[x] = max_color;
        }
    }
    let mut vertical_pixels_filled = 0;
    for (stripe_index, &stripe_color) in vertical_bg_colors.iter().enumerate() {
        if stripe_color != 16 {
            let start_x = stripe_index * 8;
            let end_x = start_x + 8;
            for y in 0..180 {
                for x in start_x..end_x {
                    if frame_buffer[x][y] == stripe_color {
                        vertical_pixels_filled += 1;
                    }
                }
            }
        }
    }

    // --- Compute horizontal stripe candidates ---
    let full_horizontal_blocks = 180 / 8; // 22
    let remainder_rows = 180 % 8; // 4
    let num_horizontal_blocks = full_horizontal_blocks + if remainder_rows > 0 { 1 } else { 0 };
    let mut horizontal_bg_colors = vec![16; num_horizontal_blocks];
    for block in 0..full_horizontal_blocks {
        let mut color_counts: AHashMap<u8, usize> = AHashMap::new();
        color_counts.insert(16, 0);
        for y in (block * 8)..((block + 1) * 8) {
            for x in 0..320 {
                let color_val = frame_buffer[x][y];
                *color_counts.entry(color_val).or_insert(0) += 1;
            }
        }
        let white_count = *color_counts.get(&16).unwrap();
        let (mut max_color, mut max_count) = (16, 0);
        for (col, count) in &color_counts {
            if *count > max_count {
                max_color = *col;
                max_count = *count;
            }
        }
        if max_color != 16 && max_count > (white_count + white_count / 16) {
            horizontal_bg_colors[block] = max_color;
        }
    }
    if remainder_rows > 0 {
        let block = full_horizontal_blocks;
        let mut color_counts: AHashMap<u8, usize> = AHashMap::new();
        color_counts.insert(16, 0);
        for y in (full_horizontal_blocks * 8)..180 {
            for x in 0..320 {
                let color_val = frame_buffer[x][y];
                *color_counts.entry(color_val).or_insert(0) += 1;
            }
        }
        let white_count = *color_counts.get(&16).unwrap();
        let (mut max_color, mut max_count) = (16, 0);
        for (col, count) in &color_counts {
            if *count > max_count {
                max_color = *col;
                max_count = *count;
            }
        }
        if max_color != 16 && max_count > (white_count + white_count / 16) {
            horizontal_bg_colors[block] = max_color;
        }
    }
    let mut horizontal_pixels_filled = 0;
    for (stripe_index, &stripe_color) in horizontal_bg_colors.iter().enumerate() {
        if stripe_color != 16 {
            let start_y = stripe_index * 8;
            let end_y = start_y + 8;
            if end_y >= 180 {
                break;
            }
            for x in 0..320 {
                for y in start_y..end_y {
                    if frame_buffer[x][y] == stripe_color {
                        horizontal_pixels_filled += 1;
                    }
                }
            }
        }
    }

    // --- Now choose and perform the better stripe–filling strategy ---
    let mut output = String::new();
    let mut total_frames = 0;
    let mut drawn_pixels: AHashMap<(i16, i16), bool> = AHashMap::new();
    let mut current_x: i16 = 0;
    let mut current_y: i16 = 0;
    let mut prev_cmd = 0;
    let mut current_color = 0;
    let mut num_to_draw = 320 * 180;

    if vertical_pixels_filled >= horizontal_pixels_filled {
        let need_any_fill = vertical_bg_colors.iter().any(|&col| col != 16);
        if need_any_fill {
            println!("Vertical stripes will fill {} pixels.", vertical_pixels_filled);
            // Use vertical stripes
            output.push_str(&format!("{{R}} {}\n", frames_per_command));
            output.push_str(&format!("{{R1}} {}\n", frames_per_command));
            output.push_str(&format!("{{R}} {}\n", frames_per_command));
            output.push_str(&format!("{{R1}} {}\n", frames_per_command));
            output.push_str(&format!("{{R}} {}\n", frames_per_command));
            output.push_str(&format!("{{R1}} {}\n", frames_per_command));
            output.push_str(&format!("{{R}} {}\n", frames_per_command));
            prev_cmd = C_R;
            let mut need_x = 0;
            current_x = 4;
            for x in 0..40 {
                let color_val = vertical_bg_colors[x];
                if color_val != 16 {
                    let color_delta = utils::get_color_dist(current_color, color_val);
                    let color_change_commands =
                        commands::get_color_change_commands(color_delta, prev_cmd);
                    for cmd in color_change_commands {
                        if (need_x > 0) && ((prev_cmd & C_R) == 0) {
                            prev_cmd = cmd | C_R;
                            need_x -= 1;
                            current_x += 1;
                        } else {
                            prev_cmd = cmd;
                        }
                        output.push_str(&format!(
                            "{{{}}} {}\n",
                            commands::cmd_to_string(prev_cmd),
                            frames_per_command
                        ));
                        total_frames += frames_per_command;
                    }
                    current_color = color_val;
                    while need_x > 0 {
                        if prev_cmd & C_R == 0 {
                            prev_cmd = C_R;
                            need_x -= 1;
                            current_x += 1;
                        } else {
                            prev_cmd = 0;
                        }
                        output.push_str(&format!(
                            "{{{}}} {}\n",
                            commands::cmd_to_string(prev_cmd),
                            frames_per_command
                        ));
                        total_frames += frames_per_command;
                    }
                    output.push_str(&format!(
                        "{{{}}} {}\n",
                        commands::cmd_to_string(C_A),
                        frames_per_command
                    ));
                    if current_y < 90 {
                        // Alternate vertical movement: down then up.
                        output.push_str("{A} (128 255) 90\n");
                        current_y = 179;
                    } else {
                        output.push_str("{A} (128 0) 90\n");
                        current_y = 0;
                    }
                    prev_cmd = C_A;
                    total_frames += frames_per_command + 90;
                }
                need_x += 8;
            }
            output.push_str(&format!("{{L1}} {}\n", frames_per_command));
            output.push_str(&format!("{{}} {}\n", frames_per_command));
            output.push_str(&format!("{{L1}} {}\n", frames_per_command));
            output.push_str(&format!("{{}} {}\n", frames_per_command));
            output.push_str(&format!("{{L1}} {}\n", frames_per_command));
            output.push_str(&format!("{{}} {}\n", frames_per_command));
            prev_cmd = 0;

        }
        // Mark drawn pixels for vertical stripes.
        for x_outer in 0..40 {
            for x_inner in 0..8 {
                let x = x_outer * 8 + x_inner;
                for y in 0..180 {
                    if frame_buffer[x][y] == vertical_bg_colors[x_outer] {
                        drawn_pixels.insert((x as i16, y as i16), true);
                        num_to_draw -= 1;
                    } else {
                        drawn_pixels.insert((x as i16, y as i16), false);
                    }
                }
            }
        }
    } else {
        println!(
            "Horizontal stripes will fill {} pixels.",
            horizontal_pixels_filled
        );
        // Use horizontal stripes
        let need_any_fill = horizontal_bg_colors.iter().any(|&col| col != 16);
        if need_any_fill {
            output.push_str(&format!("{{D}} {}\n", frames_per_command));
            output.push_str(&format!("{{R1}} {}\n", frames_per_command));
            output.push_str(&format!("{{D}} {}\n", frames_per_command));
            output.push_str(&format!("{{R1}} {}\n", frames_per_command));
            output.push_str(&format!("{{D}} {}\n", frames_per_command));
            output.push_str(&format!("{{R1}} {}\n", frames_per_command));
            output.push_str(&format!("{{D}} {}\n", frames_per_command));
            prev_cmd = C_D;
            let mut need_y = 0;
            current_y = 4;
            let mut need_pen = 3;
            let mut current_pen = 3;
            for (block, &color_val) in horizontal_bg_colors.iter().enumerate() {
                if color_val != 16 {
                    if block >= full_horizontal_blocks {
                        need_pen = 2;
                    }
                    let color_delta = utils::get_color_dist(current_color, color_val);
                    let color_change_commands =
                        commands::get_color_change_commands(color_delta, prev_cmd);
                    for cmd in color_change_commands {
                        if (need_y > 0) && ((prev_cmd & C_D) == 0) {
                            prev_cmd = cmd | C_D;
                            need_y -= 1;
                            current_y += 1;
                        } else {
                            prev_cmd = cmd;
                        }
                        output.push_str(&format!(
                            "{{{}}} {}\n",
                            commands::cmd_to_string(prev_cmd),
                            frames_per_command
                        ));
                        total_frames += frames_per_command;
                    }
                    current_color = color_val;
                    while need_y > 0 {
                        if prev_cmd & C_D == 0 {
                            prev_cmd = C_D;
                            need_y -= 1;
                            current_y += 1;
                        } else {
                            prev_cmd = 0;
                        }
                        if (need_pen < current_pen) && (prev_cmd == 0) {
                            output.push_str(&format!("{{L1}} {}\n", frames_per_command));
                            current_pen -= 1;
                        } else {
                            output.push_str(&format!(
                                "{{{}}} {}\n",
                                commands::cmd_to_string(prev_cmd),
                                frames_per_command
                            ));
                        }
                        total_frames += frames_per_command;
                    }
                    output.push_str(&format!(
                        "{{{}}} {}\n",
                        commands::cmd_to_string(C_A),
                        frames_per_command
                    ));
                    if current_x < 160 {
                        output.push_str("{A} (255 128) 160\n");
                        current_x = 319;
                    } else {
                        output.push_str("{A} (0 128) 160\n");
                        current_x = 0;
                    }
                    prev_cmd = C_A;
                    total_frames += frames_per_command + 160;
                }
                need_y += if block < (full_horizontal_blocks - 1) {
                    8
                } else {
                    6
                };
            }
            output.push_str(&format!("{{L1}} {}\n", frames_per_command));
            output.push_str(&format!("{{}} {}\n", frames_per_command));
            output.push_str(&format!("{{L1}} {}\n", frames_per_command));
            output.push_str(&format!("{{}} {}\n", frames_per_command));
            if current_pen == 3 {
                output.push_str(&format!("{{L1}} {}\n", frames_per_command));
                output.push_str(&format!("{{}} {}\n", frames_per_command));
            }
            prev_cmd = 0;
        }
        for (block, &stripe_color) in horizontal_bg_colors.iter().enumerate() {
            let (start_y, end_y) = if block < full_horizontal_blocks {
                (block * 8, (block + 1) * 8)
            } else {
                (full_horizontal_blocks * 8, 180)
            };
            for y in start_y..end_y {
                for x in 0..320 {
                    if frame_buffer[x][y] == stripe_color {
                        drawn_pixels.insert((x as i16, y as i16), true);
                        num_to_draw -= 1;
                    } else {
                        drawn_pixels.insert((x as i16, y as i16), false);
                    }
                }
            }
        }
    }

    StripeResult {
        output,
        drawn_pixels,
        num_to_draw,
        current_x,
        current_y,
        current_color,
        prev_cmd,
        total_frames,
    }
}

/// Given the post–stripe state, use greedy search to generate an ordering (a “visit order”)
/// for the remaining (undrawn) pixels.
fn generate_visit_order(
    frame_buffer: &[[u8; 180]; 320],
    drawn_pixels: &mut AHashMap<(i16, i16), bool>,
    mut num_to_draw: u32,
    start_x: i16,
    start_y: i16,
    start_color: u8,
    mut prev_cmd: u8,
    length_weight: f32,
    x_weight: f32,
    y_weight: f32,
    cache: &mut CommandCache,
) -> Vec<(i16, i16)> {
    let mut visit_order: Vec<(i16, i16)> = Vec::with_capacity(320 * 180);
    let mut temp_x = start_x;
    let mut temp_y = start_y;
    let mut temp_color = start_color;
    while num_to_draw > 0 {
        let next_coord = pathfinding::greedy_coord_search(
            temp_x,
            temp_y,
            temp_color,
            prev_cmd,
            frame_buffer,
            drawn_pixels,
            cache,
            length_weight,
            x_weight,
            y_weight,
        );
        if next_coord.is_none() {
            println!("Warning: no next pixel found even though pixels remain.");
            break;
        }
        let next_coord = next_coord.unwrap();
        visit_order.push(next_coord);
        drawn_pixels.insert(next_coord, true);
        let next_color = frame_buffer[next_coord.0 as usize][next_coord.1 as usize];
        prev_cmd = cache
            .get_result(
                next_coord.0 - temp_x,
                next_coord.1 - temp_y,
                utils::get_color_dist(temp_color, next_color),
                prev_cmd,
            )
            .last()
            .copied()
            .unwrap();
        temp_x = next_coord.0;
        temp_y = next_coord.1;
        temp_color = next_color;
        num_to_draw -= 1;
    }
    visit_order
}

/// Run several optimization routines on the visit order.
fn minor_optimize_visit_order(
    visit_order: &mut Vec<(i16, i16)>
) {
    optimizer::optimize_chunks(visit_order, 80);
    optimizer::rearrange_pixels(visit_order, 10, 6);
    optimizer::rearrange_pixels(visit_order, 100, 30);
}

fn major_optimize_visit_order(
        visit_order: &mut Vec<(i16, i16)>,
        optimize: usize,
        frame_buffer: &[[u8; 180]; 320],
        cache: &mut CommandCache,
    ) {
        let presize = optimize * 4 / 5;
    println!("Optimizing with sliding window of length {}.", presize);
    let total = visit_order.len() - presize;
    let mut last_feedback_percentage = 0;
    for index in (0..total).rev() {
        optimizer::optimize_segment(visit_order, index, presize, frame_buffer, cache);
        let progress = (total - index) * 100 / total;
        if progress >= last_feedback_percentage + 20 {
            println!("Sliding window optimization {}% complete", progress);
            last_feedback_percentage = progress;
        }
    }
    println!("Optimizing with sliding window of length {}.", optimize);
    last_feedback_percentage = 0;
    let mut index = 0;
    while index < visit_order.len() - optimize {
        optimizer::optimize_segment(visit_order, index, optimize, frame_buffer, cache);
        let feedback_percentage = index * 100 / (visit_order.len() - optimize);
        if feedback_percentage >= last_feedback_percentage + 20 {
            println!(
                "Sliding window optimization {}% complete",
                feedback_percentage
            );
            last_feedback_percentage = feedback_percentage;
        }
        index += 1;
    }
    println!("Optimizing by reinserting islands.");
    optimizer::rearrange_pixels(visit_order, 30, 20);
}

/// Generate the final command output by iterating over the visit order.
fn generate_commands(
    visit_order: &[(i16, i16)],
    frame_buffer: &[[u8; 180]; 320],
    cache: &mut CommandCache,
    mut current_x: i16,
    mut current_y: i16,
    mut current_color: u8,
    mut prev_cmd: u8,
    frames_per_command: u32,
) -> (String, u32) {
    let mut output = String::new();
    let mut total_frames = 0;
    for coord in visit_order {
        let color_val = frame_buffer[coord.0 as usize][coord.1 as usize];
        let color_delta = utils::get_color_dist(current_color, color_val);
        let commands_seq = cache.get_result(
            coord.0 - current_x,
            coord.1 - current_y,
            color_delta,
            prev_cmd,
        );
        prev_cmd = commands_seq.last().copied().unwrap();
        current_x = coord.0;
        current_y = coord.1;
        current_color = color_val;
        for &cmd in commands_seq.iter() {
            output.push_str(&format!(
                "{{{}}} {}\n",
                commands::cmd_to_string(cmd),
                frames_per_command
            ));
            total_frames += frames_per_command;
        }
    }
    (output, total_frames)
}

//
// main – The high–level flow
//
fn main() {
    // 1. Parse arguments.
    let (input_file, frames_per_command, optimize) = parse_args();
    // We ignore any user–provided length weight and instead run trials
    // over the range 1.0 to 3.0 in increments of 0.1.
    
    // 2. Load image and create frame buffer.
    let img = load_image(&input_file);
    let frame_buffer = create_frame_buffer(&img);

    // 3. Apply stripe–filling (vertical or horizontal) and update state.
    let stripe_result = apply_stripes(&frame_buffer, frames_per_command);
    let base_drawn_pixels = stripe_result.drawn_pixels;
    let num_to_draw = stripe_result.num_to_draw;
    let start_x = stripe_result.current_x;
    let start_y = stripe_result.current_y;
    let start_color = stripe_result.current_color;
    let start_prev_cmd = stripe_result.prev_cmd;
    let stripe_output = stripe_result.output;
    let stripe_total_frames = stripe_result.total_frames;

    // 4. Create a shared command cache for all trials.
    let mut cache = CommandCache::new();

    // 5. Trial runs for different length weights from 1.0 to 3.0 in increments of 0.1.
    let mut best_trial_cost: u32 = u32::MAX;
    let mut best_length_weight: f32 = 1.0;
    let mut best_x_weight: f32 = 0.0;
    let mut best_y_weight: f32 = 0.0;
    let mut best_visit_order: Option<Vec<(i16, i16)>> = None;

    println!("Starting trial runs...");
    for lw in 0..=15 {
        for xw in 0..=4 {
            for yw in 0..=4 {
                let candidate_length_weight = 1.0 + (lw as f32) * 0.2;
                let candidate_x_weight = -2.0 + (xw as f32);
                let candidate_y_weight = -2.0 + (yw as f32);
        
                // Clone the drawn_pixels so that each trial has the same starting state.
                let mut trial_drawn_pixels = base_drawn_pixels.clone();
                let trial_visit_order = generate_visit_order(
                    &frame_buffer,
                    &mut trial_drawn_pixels,
                    num_to_draw,
                    start_x,
                    start_y,
                    start_color,
                    start_prev_cmd,
                    candidate_length_weight,
                    candidate_x_weight,
                    candidate_y_weight,
                    &mut cache,
                );

                // Apply minor optimizations to get a trial cost estimate.
                let mut trial_order_optimized = trial_visit_order.clone();
                minor_optimize_visit_order(&mut trial_order_optimized);

                // Generate commands to calculate cost.
                let (_, trial_frames) = generate_commands(
                    &trial_order_optimized,
                    &frame_buffer,
                    &mut cache,
                    start_x,
                    start_y,
                    start_color,
                    start_prev_cmd,
                    frames_per_command,
                );
                let candidate_total_frames = stripe_total_frames + trial_frames;
                println!("Trial with l_w {:.1} x_w {:.1} y_w {:.1} resulted in {} frames.", candidate_length_weight, candidate_x_weight, candidate_y_weight, candidate_total_frames);

                if candidate_total_frames < best_trial_cost {
                    best_trial_cost = candidate_total_frames;
                    best_length_weight = candidate_length_weight;
                    best_x_weight = candidate_x_weight;
                    best_y_weight = candidate_y_weight;
                    best_visit_order = Some(trial_visit_order);
                }
            }
        }
    }

    println!("Best weights: length {:.1}, x {:.1}, y {:.1}. Starting with {} frames.", best_length_weight, best_x_weight, best_y_weight, best_trial_cost);

    // 6. Now, using the best length weight, run the final optimization passes.
    let mut final_visit_order = best_visit_order.unwrap();
    // Apply final optimizations:
    minor_optimize_visit_order(&mut final_visit_order);
    major_optimize_visit_order(&mut final_visit_order, optimize as usize, &frame_buffer, &mut cache);
    minor_optimize_visit_order(&mut final_visit_order);

    // 7. Generate the command output for the final visit order.
    let (cmd_output, cmd_frames) = generate_commands(
        &final_visit_order,
        &frame_buffer,
        &mut cache,
        start_x,
        start_y,
        start_color,
        start_prev_cmd,
        frames_per_command,
    );
    let total_frames = stripe_total_frames + cmd_frames;

    let mut output_string = format!(
        "; calculated with parameters: o-level {}, weights L{:.1} X{:.1} Y{:.1}. \n",
        optimize, best_length_weight, best_x_weight, best_y_weight
    );
    output_string.push_str(&stripe_output);
    output_string.push_str(&cmd_output);

    // 8. Write the output to a file.
    let input_path = Path::new(&input_file);
    let total_seconds = (total_frames as f32 / 60.0).round() as u32;
    let runtime_minutes = total_seconds / 60;
    let runtime_seconds = total_seconds % 60;
    println!(
        "Drawing is approximately {}m {}s.",
        runtime_minutes, runtime_seconds
    );
    let output_filename = format!(
        "{}_{}m_{}s_{}fpc.txt",
        input_path.file_stem().unwrap().to_str().unwrap(),
        runtime_minutes,
        runtime_seconds,
        frames_per_command
    );
    
    let output_path = input_path.with_file_name(output_filename);
    let file = match std::fs::File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error creating output file: {}", e);
            return;
        }
    };
    let mut writer = BufWriter::new(file);
    if let Err(e) = writer.write_all(output_string.as_bytes()) {
        eprintln!("Error writing output file: {}", e);
        return;
    }
}
