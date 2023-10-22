// This program converts an image to a sequence of commands for the SwiCC device
// to draw as a comment in Super Mario Maker 2.

extern crate image;
use image::{GenericImageView};

use std::collections::HashMap;
use std::env;
use std::path::{Path};
use std::io::{BufWriter, Write};

// Button bitmasks
const C_A: u8 = 1<<0;
const C_L: u8 = 1<<1;
const C_R: u8 = 1<<2;
const C_U: u8 = 1<<3;
const C_D: u8 = 1<<4;
const C_L2: u8 = 1<<5;
const C_R2: u8 = 1<<6;

fn calculate_coord_cost(sequence_length: u16, x: i16, y: i16, length_weight: f32) -> u16 {
    return (sequence_length as f32 * length_weight) as u16 + (x + y) as u16;
}


fn get_color_dist(current: u8, desired: u8) -> i8 {
    let mut move_dist = desired as i8 - current as i8;
    if move_dist < -8 {
        move_dist += 17;
    }
    if move_dist > 8 {
        move_dist -= 17;
    }
    move_dist
}

// get the commands to change the color
// color_delta is the difference between the desired color and the current color
// prev_cmd is the previous command
fn get_color_change_commands(color_delta: i8, prev_cmd: u8) -> Vec<u8> {
    let mut moves = vec![];
    if color_delta == 0 {
        return moves;
    }
    let mut add_blank = prev_cmd & (C_R2 | C_L2) != 0;
    let mut need = color_delta;
    while need != 0 {
        if add_blank {
            // Moved color previous frame; need to skip a frame
            moves.push(0);
        }
        add_blank = true;
        if need > 0 { // color right
            moves.push(C_R2);
            need -= 1;
        } else { // color left
            moves.push(C_L2);
            need += 1;
        }
    }
    moves
}

// function to translate command bitmask into string
fn cmd_to_string(cmd: u8) -> String {
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

// finds the closest pixel to the given pixel that has not been drawn yet
fn closest_undrawn_points(x: i16, y: i16, max_dist: i16, min_count: i16, drawn_pixels: &HashMap<(i16, i16), bool>) -> Vec<(i16, i16)> {
    let max_dist = if max_dist>320 {320} else {max_dist};
    let mut points = vec![];
    let mut got_count = 0;

    for d in 1..max_dist {
        for i in 0..d {
            let test_points = [
                (x+d-i, y+i),
                (x-i, y+d-i),
                (x-d+i, y-i),
                (x+i, y-d+i),
            ];

            for &(nx, ny) in &test_points {
                if !drawn_pixels.get(&(nx, ny)).unwrap_or(&true) {
                    points.push((nx, ny));
                    got_count += 1;
                }
            }
        }
        if got_count >= min_count {
            break;
        }
    }
    points
}


#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct CacheKey {
    pixel_offset_x: i16,
    pixel_offset_y: i16,
    color_delta: i8,
    prev_cmd: u8,
}

// The value stored in the command cache.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct CacheValue {
    result: Vec<u8>,
}

// The command cache.
#[derive(Debug, Clone)]
struct CommandCache {
    cache: HashMap<CacheKey, CacheValue>,
    cost_cache: HashMap<CacheKey, (u16,u8)>,
}

// Functionality to store and retrieve command sequences from the cache.
impl CommandCache {
    fn new() -> Self {
        CommandCache {
            cache: HashMap::new(),
            cost_cache: HashMap::new(),
        }
    }

    fn get_result(&mut self, pixel_offset_x: i16, pixel_offset_y: i16, color_delta: i8, prev_cmd: u8) -> Vec<u8> {
        let key = CacheKey {
            pixel_offset_x,
            pixel_offset_y,
            color_delta,
            prev_cmd,
        };

        // If the distance is small enough and result is in the cache, clone it and return it.
        if std::cmp::max(pixel_offset_x.abs(), pixel_offset_y.abs()) < 16 {
            if let Some(value) = self.cache.get(&key) {
                return value.result.clone();
            }
            // Otherwise, compute the result, store it in the caches, and return it.
            let result = derive_commands(pixel_offset_x, pixel_offset_y, color_delta, prev_cmd);
            let value = CacheValue { result: result.clone() };
            let cost = (result.len() as u16, result.last().unwrap_or(&0).clone());
            self.cache.insert(key.clone(), value);
            self.cost_cache.insert(key, cost);

            return result;
        }

        // If distance is too large, just return computed sequence without chaching.
        return derive_commands(pixel_offset_x, pixel_offset_y, color_delta, prev_cmd);
 
    }

    fn get_cost(&mut self, pixel_offset_x: i16, pixel_offset_y: i16, color_delta: i8, prev_cmd: u8) -> (u16,u8) {
        let key = CacheKey {
            pixel_offset_x,
            pixel_offset_y,
            color_delta,
            prev_cmd,
        };

        // If the distance is small enough and result is in the cache, clone it and return it.
        if std::cmp::max(pixel_offset_x.abs(), pixel_offset_y.abs()) < 16 {
            if let Some(cost) = self.cost_cache.get(&key) {
                return *cost;
            }
            // Otherwise, compute the result, store it in the caches, and return it.
            let result = derive_commands(pixel_offset_x, pixel_offset_y, color_delta, prev_cmd);
            let value = CacheValue { result: result.clone() };
            let cost = (result.len() as u16, result.last().unwrap_or(&0).clone());
            self.cache.insert(key.clone(), value);
            self.cost_cache.insert(key, cost);

            return cost;
        }

        // If distance is too large, just return computed cost without chaching.
        let sequence = derive_commands(pixel_offset_x, pixel_offset_y, color_delta, prev_cmd);
        return (sequence.len() as u16, sequence.last().unwrap_or(&0).clone());
    }
}

// Derive the commands to get to and draw a pixel.
// pixel_offset_x and pixel_offset_y are the offset of the pixel from the previous pixel.
// color_delta is the difference between the desired color and the current color.
// prev_cmd is the previous command.
fn derive_commands(pixel_offset_x: i16, pixel_offset_y: i16, color_delta: i8, prev_cmd: u8) -> Vec<u8> {
    let move_commands: Vec<u8> = get_move_commands(pixel_offset_x, pixel_offset_y, prev_cmd);
    let color_commands: Vec<u8> = get_color_change_commands(color_delta, prev_cmd);
    // Combine the commands into a single command list.
    let mut full_commands: Vec<u8> = elementwise_or(&move_commands, &color_commands);
    // Check if C_A is in the previous frame and there is only one new frame.
    if (full_commands.len() == 1) && (prev_cmd & C_A != 0) {
        // If C_L2 or C_R2 is in the new frame, need to release A otherwise it changes the previous pixel color.
        if full_commands[0] & (C_L2 | C_R2) != 0 {
            full_commands.push(0);
        }
    }
    // Add C_A to last frame
    if full_commands.len() == 0 {
        full_commands.push(C_A);
    } else {
        let final_index = full_commands.len() - 1;
        full_commands[final_index] += C_A;
    }

    full_commands
}

// Returns the bitwise OR of two vectors, padding with 0s if necessary.
fn elementwise_or(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(std::cmp::max(a.len(), b.len()));

    for i in 0..std::cmp::max(a.len(), b.len()) {
        let a_value = if i < a.len() { a[i] } else { 0 };
        let b_value = if i < b.len() { b[i] } else { 0 };
        result.push(a_value | b_value);
    }

    result
}

// Returns the commands to move to the pixel.
// d_x and d_y are the position offset.
// prev_cmd is the previous command.
fn get_move_commands(d_x: i16, d_y: i16, prev_cmd: u8) -> Vec<u8> {
    let mut commands = Vec::new();
    let mut d_x = d_x;
    let mut potential_x = 0;
    let mut d_y = d_y;
    let mut potential_y = 0;
    let mut last_cmd = prev_cmd;

    while d_x != 0 || d_y != 0 {
        let (mut potential_cmd_x, mut potential_cmd_y) = (0, 0);
        let mut progress = false;

        // Check if need to move in x-direction
        if (d_x > 0) && ((C_R & last_cmd) == 0) {
            potential_cmd_x = C_R;
            potential_x = 1;
        } else if (d_x < 0) && ((C_L & last_cmd) == 0) {
            potential_cmd_x = C_L;
            potential_x = -1;
        }

        // Check if need to move in y-direction
        if (d_y < 0) && ((C_U & last_cmd) == 0){
            potential_cmd_y = C_U;
            potential_y = -1;
        } else if (d_y > 0) && ((C_D & last_cmd) == 0) {
            potential_cmd_y = C_D;
            potential_y = 1;
        }

        if d_y.abs() > d_x.abs() {
            // Move in y-direction first if needed
            if potential_cmd_y != 0 {
                commands.push(potential_cmd_y);
                d_y -= potential_y;
                last_cmd = potential_cmd_y;
                progress = true;
            }
            // Then in x-direction
            if potential_cmd_x != 0 {
                commands.push(potential_cmd_x);
                d_x -= potential_x;
                last_cmd = potential_cmd_x;
                progress = true;
            }
        } else {
            // Move in x-direction first if needed
            if potential_cmd_x != 0 {
                commands.push(potential_cmd_x);
                d_x -= potential_x;
                last_cmd = potential_cmd_x;
                progress = true;
            }
            // Then in y-direction
            if potential_cmd_y != 0 {
                commands.push(potential_cmd_y);
                d_y -= potential_y;
                last_cmd = potential_cmd_y;
                progress = true;
            }
        }
        if !progress {
            // No progress made; need to queue a blank command
            commands.push(0);
            last_cmd = 0;
        }
    }

    commands
}

fn get_next_coord(current_x: i16, current_y: i16, current_color: u8, prev_cmd: u8, frame_buffer: &[[u8; 180]; 320], drawn_pixels: &HashMap<(i16, i16), bool>, cache: &mut CommandCache, length_weight: f32) -> Option<(i16, i16)> {
    // Search only nearby first.
    let potential_coords = closest_undrawn_points(current_x, current_y, 5, 100, drawn_pixels);
    // If no points found, expand search area.
    let potential_coords = if potential_coords.len() == 0 { closest_undrawn_points(current_x, current_y, 16, 100, drawn_pixels) } else { potential_coords };
    // If no points found, search entire image.
    let potential_coords = if potential_coords.len() == 0 { closest_undrawn_points(current_x, current_y, 320, 200, drawn_pixels) } else { potential_coords };
    // If no points found, no points left to draw.
    if potential_coords.len() == 0 {
        return None;
    }

    // Initialize the best point and cost to be the first point.
    let mut best_point = potential_coords[0];
    let color_delta = get_color_dist(current_color, frame_buffer[best_point.0 as usize][best_point.1 as usize]);
    let mut sequence_length = cache.get_cost(best_point.0 - current_x, best_point.1 - current_y, color_delta, prev_cmd).0;
    let mut min_cost = calculate_coord_cost(sequence_length, best_point.0, best_point.1, length_weight);

    // For each undrawn point...
    for &(x, y) in &potential_coords {
        // Calculate the color delta and previous command based on your specific scenario.
        let color_delta = get_color_dist(current_color, frame_buffer[x as usize][y as usize]);
        // Get the command sequence from the cache (this will also calculate and cache the sequence if it's not already cached).
        sequence_length = cache.get_cost(x - current_x, y - current_y, color_delta, prev_cmd).0;
        // Calculate the cost.
        let cost = calculate_coord_cost(sequence_length, x, y, length_weight);

        // If the cost of reaching this point is less than the cost of reaching the best point found so far...
        if cost < min_cost {
            // Update the best point and minimum cost.
            best_point = (x, y);
            min_cost = cost;
        }
    }

    // Return the best point found.
    Some(best_point)

}

fn rearrange_pixels(visit_order: &mut Vec<(i16, i16)>, isolated_block_size: usize, isolation_distance: i16) {

    let mut isolated_blocks: Vec<Vec<(i16, i16)>> = Vec::new();

    let mut i = 0;
    while i < visit_order.len() {
        let start_index = i;
        let mut end_index = i;
        let (mut prev_x, mut prev_y) = visit_order[i];

        // Find the end index of the isolated block
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

        // Check if the isolated block meets the criteria
        if block_size <= isolated_block_size {
            let block = visit_order.drain(start_index..=end_index).collect();
            isolated_blocks.push(block);
            i = start_index; // Start over from the same index since we removed elements
        } else {
            i = end_index + 1; // Continue from the next index
        }
    }

    // Reinsert the isolated blocks near existing coordinates
    for block in isolated_blocks {
        let nearest_index = find_nearest_index(&visit_order, &block[0]);
        visit_order.splice(nearest_index..nearest_index, block);
    }
}

fn find_nearest_index(visit_order: &Vec<(i16, i16)>, coord: &(i16, i16)) -> usize {
    let mut nearest_index = 0;
    let mut min_distance = i16::MAX;

    for (j, &(visit_x, visit_y)) in visit_order.iter().enumerate() {
        let distance = std::cmp::max((coord.0 - visit_x).abs(), (coord.1 - visit_y).abs());
        
        if j < visit_order.len() - 1 {
            let next_coord = visit_order[j + 1];
            let next_distance = std::cmp::max((next_coord.0 - visit_x).abs(), (next_coord.1 - visit_y).abs());
            if distance < min_distance && next_distance >= 2 {
                min_distance = distance;
                nearest_index = j;
            }
        }
    }
    
    nearest_index
}



fn generate_and_search_permutations(
    current_permutation: &mut Vec<(i16, i16)>,
    remaining_coords: &[(i16, i16)],
    current_cost: u16,
    optimal_cost: &mut u16,
    optimal_permutation: &mut Option<Vec<(i16, i16)>>,
    start_coord: (i16, i16),
    end_coord: (i16, i16),
    frame_buffer: &[[u8; 180]; 320],
    cache: &mut CommandCache,
    prev_cmd: u8,
) {
    // Calculate the lower bound of the cost for the remaining coordinates
    let lower_bound = (remaining_coords.len() as f32 * 1.5) as u16;

    // If the lower bound of the cost added to the current cost is already greater than the optimal cost, prune this branch
    if current_cost + lower_bound >= *optimal_cost {
        return;
    }

    // If there are no remaining coordinates, calculate the cost for moving from the last middle coordinate to the end coordinate, and update the optimal cost and permutation if necessary
    if remaining_coords.is_empty() {
        let last_middle_coord = *current_permutation.last().unwrap();
        let temp_color = frame_buffer[last_middle_coord.0 as usize][last_middle_coord.1 as usize];
        let last_color = frame_buffer[end_coord.0 as usize][end_coord.1 as usize];
        let last_commands = cache.get_cost(
            end_coord.0 - last_middle_coord.0,
            end_coord.1 - last_middle_coord.1,
            get_color_dist(temp_color, last_color),
            prev_cmd,
        );
        let cost = current_cost + last_commands.0;

        if cost < *optimal_cost {
            *optimal_cost = cost;
            *optimal_permutation = Some(current_permutation.clone());
        }

        return;
    }

    // For each remaining coordinate, add it to the current permutation, calculate the new cost, and recurse
    for (index, &coord) in remaining_coords.iter().enumerate() {
        let mut new_remaining_coords = remaining_coords.to_vec();
        new_remaining_coords.remove(index);
        
        let temp_x = if current_permutation.is_empty() { start_coord.0 } else { current_permutation.last().unwrap().0 };
        let temp_y = if current_permutation.is_empty() { start_coord.1 } else { current_permutation.last().unwrap().1 };
        let temp_color = frame_buffer[temp_x as usize][temp_y as usize];
        let next_color = frame_buffer[coord.0 as usize][coord.1 as usize];

        let command_cost = cache.get_cost(
            coord.0 - temp_x,
            coord.1 - temp_y,
            get_color_dist(temp_color, next_color),
            prev_cmd,
        );

        let new_cost = current_cost + command_cost.0;
        let new_prev_cmd = command_cost.1;
        
        current_permutation.push(coord);

        generate_and_search_permutations(
            current_permutation,
            &new_remaining_coords,
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
    }
}

fn optimize_segment(
    visit_order: &mut Vec<(i16, i16)>,
    index: usize,
    length: usize,
    frame_buffer: &[[u8; 180]; 320],
    cache: &mut CommandCache,
) {
    if length <= 3 { // This won't change anything.
        return;
    }
    let start_coord = visit_order[index];
    let end_coord = visit_order[index + length - 1];
    let middle_coords = visit_order[index + 1..index + length - 1].to_vec();

    // Get previous-previous command
    let prev_prev_cmd = if index < 1 {
        0
    } else {
        let prev_coord = visit_order[index];
        let prev_prev_coord = visit_order[index - 1];
        let prev_color = frame_buffer[prev_coord.0 as usize][prev_coord.1 as usize];
        let prev_prev_color = frame_buffer[prev_prev_coord.0 as usize][prev_prev_coord.1 as usize];
        let commands = cache.get_cost(
            prev_coord.0 - prev_prev_coord.0,
            prev_coord.1 - prev_prev_coord.1,
            get_color_dist(prev_prev_color, prev_color),
            C_A, //constant
        );
        commands.1
    };

    let mut optimal_permutation: Option<Vec<(i16, i16)>> = None;
    let mut optimal_cost = u16::MAX;
    let mut current_permutation = vec![];

    generate_and_search_permutations(
        &mut current_permutation,
        &middle_coords,
        0,
        &mut optimal_cost,
        &mut optimal_permutation,
        start_coord,
        end_coord,
        frame_buffer,
        cache,
        prev_prev_cmd,
    );

    // Update the visit_order with the optimal permutation
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



fn optimize_chunks(
    visit_order: &mut Vec<(i16, i16)>,
    gap_distance: i16,
) {
    // Break visit_order into chunks
    let mut chunks: Vec<Vec<(i16, i16)>> = Vec::new();
    let mut current_chunk: Vec<(i16, i16)> = Vec::new();
    for i in 0..visit_order.len() {
        current_chunk.push(visit_order[i]);
        if i < visit_order.len() - 1 {
            let curr_pixel = visit_order[i];
            let next_pixel = visit_order[i + 1];
            let distance = std::cmp::max((curr_pixel.0 - next_pixel.0).abs(), (curr_pixel.1 - next_pixel.1).abs());
            if distance > gap_distance {
                chunks.push(current_chunk.clone());
                current_chunk.clear();
            }
        }
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    // Reassemble visit_order using a greedy algorithm
    let mut new_visit_order: Vec<(i16, i16)> = chunks.remove(0);
    while !chunks.is_empty() {
        let end_coord = new_visit_order.last().unwrap().clone();
        let mut min_distance = i16::MAX;
        let mut min_index = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            let start_coord = chunk[0];
            let distance = std::cmp::max((start_coord.0 - end_coord.0).abs(), (start_coord.1 - end_coord.1).abs());
            if distance < min_distance {
                min_distance = distance;
                min_index = i;
            }
        }
        let next_chunk = chunks.remove(min_index);
        new_visit_order.extend(next_chunk);
    }

    // Update the visit_order with the new order
    *visit_order = new_visit_order;
}

fn find_closest_color(color: (u8, u8, u8), palette: &[(u8, u8, u8)]) -> u8 {
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

fn color_distance_squared(c1: (u8, u8, u8), c2: (u8, u8, u8)) -> u32 {
    let r_diff = c1.0 as i32 - c2.0 as i32;
    let g_diff = c1.1 as i32 - c2.1 as i32;
    let b_diff = c1.2 as i32 - c2.2 as i32;

    (r_diff * r_diff + g_diff * g_diff + b_diff * b_diff) as u32
}

const PALETTE: &[(u8, u8, u8)] = &[
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


fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <input_file> <frames_per_command> (optional, default 2)", args[0]);
        return;
    }
    let input_file = &args[1];
    let frames_per_command = if args.len() >= 3 { args[2].parse().unwrap_or(2) } else { 2 };
    let optimize = if args.len() >= 4 { args[3].parse().unwrap_or(6) } else { 6 };
    let length_weight = if args.len() >= 5 { args[4].parse().unwrap_or(1.8) } else { 1.8 };
    
    let img = match image::open(input_file) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Error opening file: {}", e);
            return;
        }
    };

    let (width, height) = img.dimensions();

    const TARGET_WIDTH: u32 = 320;
    const TARGET_HEIGHT: u32 = 180;
    const FILL_COLOR_INDEX: u8 = 16;
    // Create and initialize frame buffer (contains indexed colors)
    let mut frame_buffer = [[0u8; 180]; 320];
    for x in 0..TARGET_WIDTH {
        for y in 0..TARGET_HEIGHT {
            frame_buffer[x as usize][y as usize] = FILL_COLOR_INDEX;
        }
    }    

    // Calculate the starting offsets for the source image and the frame buffer
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
    
    // Iterate over the target region, adjusting for the offsets
    for x in fb_offset_x..(TARGET_WIDTH - fb_offset_x) {
        for y in fb_offset_y..(TARGET_HEIGHT - fb_offset_y) {
            let src_x = x + src_offset_x - fb_offset_x;
            let src_y = y + src_offset_y - fb_offset_y;
    
            if src_x < width && src_y < height {
                let rgba = img.get_pixel(src_x, src_y);
                let (r, g, b, _) = (rgba[0], rgba[1], rgba[2], rgba[3]);  // Ignore alpha channel
                let closest_color_index = find_closest_color((r, g, b), &PALETTE);
                frame_buffer[x as usize][y as usize] = closest_color_index;
            } else {
                frame_buffer[x as usize][y as usize] = FILL_COLOR_INDEX;
            }
        }
    }
                
    // Determine most common background color for each set of eight columns.
    let mut bg_colors: [u8; 40] = [16; 40]; // Default to white
    for x in 0..40 { // Image is 320 wide, so 40 sets of 8 columns
        let mut color_counts: HashMap<u8, usize> = HashMap::new();
        color_counts.insert(16, 0); // Make sure white is in map
        // Check each pixel in the 8-column block
        for x_inner in (x*8)..((x+1)*8) {
            for y in 0..180 {
                let color = frame_buffer[x_inner][y];
                let count = color_counts.entry(color).or_insert(0);
                *count += 1;
            }
        }
        // Only change bg color if it will outweigh the destroyed white pixels.
        let mut max_count = 0;
        let mut max_index = 0;
        let white_count = color_counts[&16];
        for (index, count) in color_counts {
            if count > max_count {
                max_count = count;
                max_index = index;
            }
        }
        if max_count > (white_count + white_count / 16)  {
            bg_colors[x] = max_index;
        }
    }

    // Keep track of drawing
    let mut total_frames = 0;
    let mut current_x: i16 = 0;
    let mut current_y: i16 = 0;
    let mut prev_cmd = 0;
    let mut current_color = 0;
    let mut num_to_draw = 320*180;

    // Create output string
    let mut output_string = String::new();
    output_string.push_str(&format!("; calculated with parameters: o-level {}, length weight {}\n", optimize, length_weight));


    // Add initial commands to the output string to fill any column blocks that aren't white.
    let need_any_fill = bg_colors.iter().any(|&x| x != 16);
    if need_any_fill {
        let mut going_down = true;
        // Add initial pen size change and move right four pixels.
        output_string.push_str(&format!("{{R}} {}\n", frames_per_command));
        output_string.push_str(&format!("{{R1}} {}\n", frames_per_command));
        output_string.push_str(&format!("{{R}} {}\n", frames_per_command));
        output_string.push_str(&format!("{{R1}} {}\n", frames_per_command));
        output_string.push_str(&format!("{{R}} {}\n", frames_per_command));
        output_string.push_str(&format!("{{R1}} {}\n", frames_per_command));
        output_string.push_str(&format!("{{R}} {}\n", frames_per_command));
        prev_cmd = C_R;
        let mut need_x = 0;
        current_x = 4;
        // Go over each column block
        for x in 0..40 {
            let color = bg_colors[x];
            if color != 16 { // Not white
                let color_delta = get_color_dist(current_color, color);
                let color_change_commands = get_color_change_commands(color_delta, prev_cmd);
                for cmd in color_change_commands {
                    // While changing color, also move right if needed.
                    if (need_x > 0) && ((prev_cmd & C_R) == 0) {
                        prev_cmd = cmd | C_R;
                        need_x -= 1;
                        current_x += 1;
                    } else {
                        prev_cmd = cmd;
                    }
                    // Add button string enclosed in curly brackets to output string.
                    output_string.push_str(&format!("{{{}}} {}\n", cmd_to_string(prev_cmd), frames_per_command));
                    total_frames += frames_per_command;
                }
                current_color = color;
                // If still need to move right, do so, ensuring that there are not consecutive C_R commands.
                while need_x > 0 {
                    if prev_cmd & C_R == 0 {
                        prev_cmd = C_R;
                        need_x -= 1;
                        current_x += 1;
                    } else {
                        prev_cmd = 0;
                    }
                    output_string.push_str(&format!("{{{}}} {}\n", cmd_to_string(prev_cmd), frames_per_command));
                    total_frames += frames_per_command;
                }

                // Press A to begin painting
                output_string.push_str(&format!("{{{}}} {}\n", cmd_to_string(C_A), frames_per_command));
                // Hold A while moving cursor up or down for 90 frames.
                if going_down {
                    output_string.push_str(&"{A} (128 255) 90\n");
                    current_y = 179;
                } else {
                    output_string.push_str(&"{A} (128 0) 90\n");
                    current_y = 0;
                }
                prev_cmd = C_A;
                total_frames += frames_per_command + 90;
                going_down = !going_down;
            }
            // Going to next block requires x offset 8
            need_x += 8;
        }
        // Put pen back small
        output_string.push_str(&format!("{{L1}} {}\n", frames_per_command));
        output_string.push_str(&format!("{{}} {}\n", frames_per_command));
        output_string.push_str(&format!("{{L1}} {}\n", frames_per_command));
        output_string.push_str(&format!("{{}} {}\n", frames_per_command));
        output_string.push_str(&format!("{{L1}} {}\n", frames_per_command));
        output_string.push_str(&format!("{{}} {}\n", frames_per_command));
        prev_cmd = 0;
    }


    // Create a HashMap with the x,y coordinate as the key and whether or not it has been drawn as the value.
    let mut drawn_pixels: HashMap<(i16, i16), bool> = HashMap::new();
    // Check each pixel in frame_buffer and if its color is the background color for that column block, mark it as drawn.
    for x_outer in 0..40 {
        for x_inner in 0..8 {
            let x = x_outer * 8 + x_inner;
            for y in 0..180 {
                let color = frame_buffer[x][y];
                if color == bg_colors[x_outer] {
                    drawn_pixels.insert((x as i16, y as i16), true);
                    num_to_draw -= 1;
                } else {
                    drawn_pixels.insert((x as i16, y as i16), false);
                }
            }
        }
    }

    // Create a cache for known commands using CommandCache
    let mut cache = CommandCache::new();

    // Create a vector of coordinates to visit.
    let mut visit_order: Vec<(i16, i16)> = Vec::new();

    let mut temp_color = current_color;
    let mut temp_x = current_x;
    let mut temp_y = current_y;
    let mut next_print_percentage = 80;
    while num_to_draw > 0 {
        // Find the best next pixel to the current position that has not been drawn.  If None, break.
        let next_coord = get_next_coord(temp_x, temp_y, temp_color, prev_cmd, &frame_buffer, &drawn_pixels, &mut cache, length_weight);
        if next_coord.is_none() {
            println!("Warning: no next pixel found in image data even though too few pixels have been placed.");
            break;
        }
        let next_coord = next_coord.unwrap();
        // Add the next coordinate to the vector of coordinates to visit.
        visit_order.push(next_coord);
        // Mark the closest coordinate as drawn.
        drawn_pixels.insert(next_coord, true);

        // Update state
        let next_color = frame_buffer[next_coord.0 as usize][next_coord.1 as usize];
        // prev_cmd becomes last commands of cache result
        prev_cmd = *cache.get_result(next_coord.0 - temp_x, next_coord.1 - temp_y, get_color_dist(temp_color, next_color), prev_cmd).last().unwrap();
        temp_x = next_coord.0;
        temp_y = next_coord.1;
        temp_color = next_color;
        num_to_draw -= 1;
        // If percentage of pixels drawn has decreased enough, print feedback.
        let current_percentage = num_to_draw * 100 / (320*180);
        if current_percentage < next_print_percentage {
            println!("{}% remaining.", next_print_percentage);
            next_print_percentage -= 20;
        }
    }

    // Rearrange chunks of pixels.
    optimize_chunks(&mut visit_order, 30);
    // Go over the sequence of pixels to find isolated blocks and reinsert them at better locations.
    rearrange_pixels(&mut visit_order, 10, 6);
    rearrange_pixels(&mut visit_order, 36, 6);



    if optimize >= 8 {
        // Try some brute force optimization
        let length = optimize * 3 / 5;
        println!("Optimizing sequence length {}.", length);
        let mut last_feedback_percentage = 0;
        let mut index = 0;
        while index < visit_order.len() - length {
            optimize_segment(&mut visit_order, index, length, &frame_buffer, &mut cache);
            // Print feedback once in a while.
            let feedback_percentage = index * 100 / (visit_order.len() - length);
            if feedback_percentage >= (last_feedback_percentage + 20) {
                println!("Optimization {}% complete", feedback_percentage);
                last_feedback_percentage = feedback_percentage;
            }
            index += 1;
        }                
    }

    if optimize >= 4 {
        // Try some brute force optimization
        let length = optimize;
        println!("Optimizing sequence length {}.", length);
        let mut last_feedback_percentage = 0;
        let mut index = 0;
        while index < visit_order.len() - length {
            optimize_segment(&mut visit_order, index, length, &frame_buffer, &mut cache);
            // Print feedback once in a while.
            let feedback_percentage = index * 100 / (visit_order.len() - length);
            if feedback_percentage >= (last_feedback_percentage + 20) {
                println!("Optimization {}% complete", feedback_percentage);
                last_feedback_percentage = feedback_percentage;
            }
            index += length / 2;
        }                
    }

    // Go over the sequence of pixels again to find isolated blocks and reinsert them at better locations.
    rearrange_pixels(&mut visit_order, 10, 6);
    rearrange_pixels(&mut visit_order, 25, 6);

    // Convert the sequence of coordinates to a sequence of commands and add to output string.
    let mut prev_cmd = 0;
    for coord in &visit_order {
        let color = frame_buffer[coord.0 as usize][coord.1 as usize];
        let color_delta = get_color_dist(current_color, color);
        // Get command sequence for this coordinate from the cache.
        let commands = cache.get_result(coord.0 - current_x, coord.1 - current_y, color_delta, prev_cmd);
        // Update state
        prev_cmd = *commands.last().unwrap();
        current_x = coord.0;
        current_y = coord.1;
        current_color = color;
        // Add button string enclosed in curly brackets to output string.
        for cmd in commands {
            output_string.push_str(&format!("{{{}}} {}\n", cmd_to_string(cmd), frames_per_command));
            total_frames += frames_per_command;
        }
    }

    // Save the result to a text file with the same name as the input file and the time.
    let input_path = Path::new(input_file);
    let total_seconds = (total_frames as f32 / 60f32).round() as u32;
    let runtime_minutes = total_seconds / 60;
    let runtime_seconds = total_seconds % 60;
    println!("Drawing is approximately {}m {}s.", runtime_minutes, runtime_seconds);
    let output_filename = format!("{}_{}m_{}s_{}fpc.txt", 
            input_path.file_stem().unwrap().to_str().unwrap(), 
            runtime_minutes as usize,
            runtime_seconds as usize,
            frames_per_command );
    let output_path = input_path.with_file_name(output_filename);
    let file = match std::fs::File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error creating output file: {}", e);
            return;
        }
    };
    let mut writer = BufWriter::new(file);
    match writer.write_all(output_string.as_bytes()) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Error writing output file: {}", e);
            return;
        }
    }

}