use std::collections::HashMap;
use std::env;
use std::fs::{File};
use std::path::{Path};
use std::io::{BufWriter, Write};


const C_A: u8 = 1<<0;
const C_L: u8 = 1<<1;
const C_R: u8 = 1<<2;
const C_U: u8 = 1<<3;
const C_D: u8 = 1<<4;
const C_L2: u8 = 1<<5;
const C_R2: u8 = 1<<6;

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

fn get_color_change_commands(d_color: i8, prev_cmd: u8) -> Vec<u8> {
    let mut moves = vec![];
    if ((prev_cmd & C_R2) != 0) || ((prev_cmd & C_L2) != 0) {
        // Moved color previous frame; need to skip a frame
        moves.push(0);
    }
    let mut need = d_color;
    if need > 0 { // color right
        while need != 0 {
            moves.push(C_R2);
            moves.push(0);
            need -= 1;
        }
    }
    else { // color left
        while need != 0 {
            moves.push(C_L2);
            moves.push(0);
            need += 1;
        }
    }
    if moves.len() >= 2 {
        moves.pop(); // Remove trailing blank
    }
    moves
}

fn get_move_commands(d_x: isize, d_y: isize, prev_cmd: u8) -> Vec<u8> {
    let mut moves: Vec<u8> = vec![];
    let mut mdx = d_x;
    let mut mdy = d_y;
    let mut skipped: u8 = 0;
    let mut last_add = prev_cmd & (C_L | C_R | C_U | C_D);

    while mdx != 0 || mdy != 0 {
        // For each direction, add the command if needed, if possible, and if
        // that direction needs more than the other direction.
        if mdx > 0 && last_add != C_R && (mdx.abs() >= mdy.abs()) {
            moves.push(C_R);
            last_add = C_R;
            mdx -= 1;
            skipped = 0;
        } else { skipped += 1; }
        if mdx < 0 && last_add != C_L && (mdx.abs() >= mdy.abs()) {
            moves.push(C_L);
            last_add = C_L;
            mdx += 1;
            skipped = 0;
        } else { skipped += 1; }
        if mdy > 0 && last_add != C_D && (mdy.abs() >= mdx.abs()) {
            moves.push(C_D);
            last_add = C_D;
            mdy -= 1;
            skipped = 0;
        } else { skipped += 1; }
        if mdy < 0 && last_add != C_U && (mdy.abs() >= mdx.abs()) {
            moves.push(C_U);
            last_add = C_U;
            mdy += 1;
            skipped = 0;
        } else { skipped += 1; }
        // Wasn't able to move; must need a buffer
        if skipped >= 4 {
            moves.push(0);
            last_add = 0;
            skipped = 0;
        }
    }

    moves
}

fn elementwise_or(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(std::cmp::max(a.len(), b.len()));

    for i in 0..std::cmp::max(a.len(), b.len()) {
        let a_value = if i < a.len() { a[i] } else { 0 };
        let b_value = if i < b.len() { b[i] } else { 0 };
        result.push(a_value | b_value);
    }

    result
}

fn get_commands(d_x: isize, d_y: isize, d_color: i8, prev_cmd: u8) -> Vec<u8> 
{
    let move_commands: Vec<u8> = get_move_commands(d_x, d_y, prev_cmd);
    let color_commands: Vec<u8> = get_color_change_commands(d_color, prev_cmd);
    // Combine the commands into a single command list.
    let mut full_commands: Vec<u8> = elementwise_or(&move_commands, &color_commands);
    // Check if C_A is in the previous frame and there is only one new frame.
    if (full_commands.len() == 1) && (prev_cmd & C_A != 0) {
        // If C_L2 or C_R2 is in the new frame, need to release A otherwise it changes the previous pixel color.
        if full_commands[0] & (C_L2 | C_R2) != 0 {
            full_commands.push(0);
        }
    }
    // Add C_A to final frame
    if full_commands.len() == 0 {
        full_commands.push(C_A);
    } else {
        let final_index = full_commands.len() - 1;
        full_commands[final_index] += C_A;
    }

    full_commands

}

fn calculate_cost(cmd_len: usize, x_pos: usize, y_pos: usize, _start_at_end: bool) -> usize{
    /* // Option to start at bottom right after filling with a color.
    (cmd_len * 3)
        + if start_at_end {(x_pos as isize - 319 as isize).abs() as usize} else {0}
        + if start_at_end {(y_pos as isize - 179 as isize).abs() as usize} else {0}
        */
    (cmd_len * 3) + x_pos + y_pos
}

fn determine_path(block: &[Vec<u8>], bg_color: u8, cmd_cache: &mut HashMap<(isize, isize, i8, u8), Vec<u8>>, start_at_end: bool)
    -> Vec<(usize, usize)>
{

    let mut pixel_sequence: Vec<(usize, usize)> = vec![];

    let mut current_color: u8 = 0;
    let mut current_x: usize = if start_at_end {block[0].len() - 1} else {0};
    let mut current_y: usize = if start_at_end {block.len() - 1} else {0};

    let mut prev_cmd: u8 = 0;

    let mut visited_count: usize = 0;
    let mut visited_map: HashMap<(usize, usize), bool> = HashMap::new();
    // "Already visited" all pixels with the background color
    for y in 0..block.len() {
        for x in 0..block[0].len() {
            if block[y][x] == bg_color {
                visited_map.insert((x, y), true);
                visited_count += 1;
            }
        }
    }

    // Used to print feedback as the block processes
    let mut next_feedback = visited_count * 10 / (block.len() * block[0].len()) * 10 + 10;


    while visited_count < block.len() * block[0].len() {
        if visited_count * 10 / (block.len() * block[0].len()) * 10 >= next_feedback {
            println!("Path calculated for {}% of image.", next_feedback);
            next_feedback += 10;
        }

        let mut min_score = std::usize::MAX;
        let mut min_dist = std::usize::MAX;
        let mut min_command: Vec<u8> = vec![];
        let mut next_x = 0;
        let mut next_y = 0;

        // Define a smaller search area around current coordinate
        let search_min_x = usize::saturating_sub(current_x, 9);
        let search_max_x = (current_x + 9).min(block[0].len() - 1);
        let search_min_y = usize::saturating_sub(current_y, 9);
        let search_max_y = (current_y + 9).min(block.len() - 1);

        // Greedy algorithm in smaller area
        for y in search_min_y..=search_max_y {
            for x in search_min_x..=search_max_x {
                if !visited_map.contains_key(&(x, y)) {
                    // Check if in cache, follow up as needed.
                    let potential = 
                            match cmd_cache.get(&(x as isize - current_x as isize, y as isize - current_y as isize, block[y][x] as i8 - current_color as i8, prev_cmd)) {
                            Some(existing) => 
                                existing.clone(),
                            None => {
                                let d_x = x as isize - current_x as isize;
                                let d_y = y as isize - current_y as isize;
                                let d_color = get_color_dist(current_color, block[y][x]);
                                let result = get_commands(d_x, d_y, d_color, prev_cmd);
                                cmd_cache.insert((d_x, d_y, d_color, prev_cmd), result.clone());
                                result
                            }
                        };

                    let score = calculate_cost(potential.len(), x, y, start_at_end);
                    if score < min_score {
                        min_dist = std::cmp::min(min_dist, potential.len());
                        min_score = score;
                        min_command = potential.clone();
                        next_x = x;
                        next_y = y;
                    }
                }
            }
        }

        if min_dist > 16 { // Didn't find a short option nearby.
            min_score = std::usize::MAX;
            // Greedy algorithm over all pixels in block
            for y in 0..block.len() {
                for x in 0..block[0].len() {
                    if !visited_map.contains_key(&(x, y)) {
                        // Check if in cache, follow up as needed.
                        let potential = 
                            match cmd_cache.get(&(x as isize - current_x as isize, y as isize - current_y as isize, block[y][x] as i8 - current_color as i8, prev_cmd)) {
                                Some(existing) => 
                                    existing.clone(),
                                None => {
                                    let d_x = x as isize - current_x as isize;
                                    let d_y = y as isize - current_y as isize;
                                    let d_color = get_color_dist(current_color, block[y][x]);
                                    let result = get_commands(d_x, d_y, d_color, prev_cmd);
                                    cmd_cache.insert((d_x, d_y, d_color, prev_cmd), result.clone());
                                    result
                                }
                            };

                            let score = calculate_cost(potential.len(), x, y, start_at_end);
                        if score < min_score {
                            min_score = score;
                            min_command = potential.clone();
                            next_x = x;
                            next_y = y;
                        }
                    }
                }
            }
        }

        // Add minimum found to commands and update currents
        prev_cmd = min_command[min_command.len() - 1];
        pixel_sequence.push((next_x, next_y));
        current_color = block[next_y][next_x];
        current_x = next_x;
        current_y = next_y;
        // Track the visit
        visited_map.insert((next_x, next_y), true);
        visited_count += 1;

    }

    pixel_sequence
}


fn l1_norm(a: &(usize, usize), b: &(usize, usize)) -> usize {
    ((a.0 as isize - b.0 as isize).abs() + (a.1 as isize - b.1 as isize).abs()) as usize
}

fn find_best_insertion_position(new_path: &Vec<(usize, usize)>, isolated_group: &[(usize, usize)]) -> usize {
    let mut best_position = 0;
    let mut best_distance = std::usize::MAX;

    for i in 0..new_path.len() - 1 {
        let distance = l1_norm(&new_path[i], &isolated_group[0]) + l1_norm(&isolated_group[isolated_group.len() - 1], &new_path[i + 1]);
        if distance < best_distance {
            best_position = i + 1;
            best_distance = distance;
        }
    }

    best_position
}
fn optimize_path(path: Vec<(usize, usize)>, separation: usize) -> Vec<(usize, usize)> {
    let mut new_path: Vec<(usize, usize)> = vec![];
    let mut isolated_group: Vec<(usize, usize)> = vec![];
    let mut isolated_groups: Vec<Vec<(usize, usize)>> = vec![];

    for i in 0..path.len() - 1 {
        let distance = l1_norm(&path[i], &path[i + 1]);

        // Current pixel is part of this group
        isolated_group.push(path[i]);
        // Check if about to have a big jump
        if distance >= separation {
            // Check if collected group was short enough to be considered a blip
            if distance >= (isolated_group.len() * 3) {
                isolated_groups.push(isolated_group.clone());
            } else {
                new_path.extend(isolated_group.iter().cloned());
            }
            isolated_group.clear();
        }
    }

    // Handle the last group in the sequence
    isolated_group.push(path[path.len() - 1]);
    if isolated_group.len() <= separation {
        isolated_groups.push(isolated_group.clone());
    } else {
        new_path.extend(isolated_group.iter().cloned());
    }

    for group in isolated_groups {
        let best_position = find_best_insertion_position(&new_path, &group);
        new_path.splice(best_position..best_position, group.iter().cloned());
    }

    new_path
}

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <input_file> <frames_per_command> (optional, default 2)", args[0]);
        return;
    }
    let input_file = &args[1];
    let frame_count = if args.len() >= 3 { args[2].parse().unwrap_or(2) } else { 2 };

    // Open the input file
    let file = match File::open(input_file) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening file: {}", e);
            return;
        }
    };

    // Configure the decoder such that it will expand the image to indexed color.
    let mut decoder = gif::DecodeOptions::new();
    decoder.set_color_output(gif::ColorOutput::Indexed);
    // Read the file header
    let mut reader = match decoder.read_info(file) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error reading file header: {}", e);
            return;
        }
    };

    // Read the first frame
    let frame = match reader.read_next_frame() {
        Ok(Some(f)) => f,
        Ok(None) => {
            eprintln!("No frames in file.");
            return;
        }
        Err(e) => {
            eprintln!("Error reading frame: {}", e);
            return;
        }
    };

    // Verify image dimensions
    if (frame.width != 320) || (frame.height != 180) {
        eprintln!("Image wrong size. Rejected.");
        return;
    }

    // Find most common color index in the range 0-16.
    let mut color_counts = [0u32; 17];
    for pixel in frame.buffer.iter() {
        if *pixel <= 16 {
            color_counts[*pixel as usize] += 1;
        } else {
            eprintln!("Warning: invalid color index found: {}.", *pixel);
        }
    }
    let mut max_count = 0;
    let mut max_index = 0;
    for i in 0..17 {
        if color_counts[i] > max_count {
            max_count = color_counts[i];
            max_index = i;
        }
    }

    let bg_color: u8 = 
            if max_count > (320*180/12) { // Requre at least 1/12 of pixels to be that color to use as fill.
                println!("Using common color index ({}) as background.", max_index);
                max_index as u8
            } else {
                println!("Using white background.");
                16
            };

    // Reformat data into 2D array
    let image_data = frame.buffer.chunks(frame.width as usize)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();


    // This hash will save command sequences based on dx, dy, dcolor, and prev_cmd
    let mut cmd_cache: HashMap<(isize, isize, i8, u8), Vec<u8>> = HashMap::new();
    // Process the image 
    let initial_path = determine_path(&image_data, bg_color, &mut cmd_cache, bg_color!=16);
    // Try some optimization
    let optimized_path = optimize_path(initial_path, 16);
    let optimized_path_2 = optimize_path(optimized_path, 13);
    let optimized_path_3 = optimize_path(optimized_path_2, 10);
    
    // Convert to command sequence
    let mut x = if bg_color==16 {0} else {image_data[0].len() - 1};
    let mut y = if bg_color==16 {0} else {image_data.len() - 1};
    let mut current_color = 0;
    let mut previous_command: u8 = 0;
    let mut command_sequence: Vec<u8> = vec![0];
    
    for &(next_x, next_y) in &optimized_path_3 {

        let d_x = next_x as isize - x as isize;
        let d_y = next_y as isize - y as isize;
        let next_color = image_data[next_y][next_x];
        let d_color = get_color_dist(current_color, next_color);
        let key = (d_x, d_y, d_color, previous_command);
        let cmds = cmd_cache.entry(key).or_insert_with(|| {
            get_commands(d_x, d_y, d_color, previous_command)
        });
        command_sequence.append(&mut cmds.clone());
        previous_command = *cmds.last().unwrap_or(&0);
        x = next_x;
        y = next_y;
        current_color = next_color;
    }

    // Count how many frames this is going to take.
    let num_frames = command_sequence.len() * frame_count + if bg_color != 16 {3600} else {0};
    let total_seconds = (num_frames as f32 / 60f32).round() as u32;

    // Save the result to a text file with the same name as the input file and the time.
    let input_path = Path::new(input_file);
    let runtime_minutes = total_seconds / 60;
    let runtime_seconds = total_seconds % 60;
    println!("Drawing is approximately {}:{}.", runtime_minutes, runtime_seconds);
    let output_filename = format!("{}_{}m_{}s_{}fpc.txt", 
            input_path.file_stem().unwrap().to_str().unwrap(), 
            runtime_minutes as usize,
            runtime_seconds as usize,
            frame_count );
    let output_path = input_path.with_file_name(output_filename);
    let file = match std::fs::File::create(&output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error creating output file: {}", e);
            return;
        }
    };
    let mut writer = BufWriter::new(file);

    // If filling background, write those commands.
    if bg_color != 16 {
        // Set color
        for _ in 0..bg_color {
            writeln!(writer, "{{R2}} {}", frame_count).unwrap();
            writeln!(writer, "{{}} {}", frame_count).unwrap();
        }
        // Set large cursor and move down 3 pixels
        for _ in 0..3 {
            writeln!(writer, "{{D}} {}", frame_count).unwrap();
            writeln!(writer, "{{R1}} {}", frame_count).unwrap();
        }
        writeln!(writer, "{{A}} {}", frame_count).unwrap();
        for _ in 0..11 {
            writeln!(writer, "{{A}} 170 255").unwrap();
            for _ in 0..8 {
                writeln!(writer, "{{A D}} {}", frame_count).unwrap();
                writeln!(writer, "{{A}} {}", frame_count).unwrap();
            }
            writeln!(writer, "{{A}} 170 1").unwrap();
            for _ in 0..8 {
                writeln!(writer, "{{A D}} {}", frame_count).unwrap();
                writeln!(writer, "{{A}} {}", frame_count).unwrap();
            }
        }
        writeln!(writer, "{{A}} 170 255").unwrap();
        // Reset color
        for _ in 0..bg_color {
            writeln!(writer, "{{L2}} {}", frame_count).unwrap();
            writeln!(writer, "{{}} {}", frame_count).unwrap();
        }
        // Reset cursor
        for _ in 0..3 {
            writeln!(writer, "{{L1}} {}", frame_count).unwrap();
            writeln!(writer, "{{}} {}", frame_count).unwrap();
        }
        // Return home
        writeln!(writer, "{{}} 4 255 255").unwrap();

    }

    // Write the commands.
    for line in command_sequence {
        let mut frame_string = String::new();
        if line & C_L != 0 {
            frame_string += "L ";
        }
        if line & C_R != 0 {
            frame_string += "R ";
        }
        if line & C_U != 0 {
            frame_string += "U ";
        }
        if line & C_D != 0 {
            frame_string += "D ";
        }
        if line & C_L2 != 0 {
            frame_string += "L2 ";
        }
        if line & C_R2 != 0 {
            frame_string += "R2 ";
        }
        if line & C_A != 0 {
            frame_string += "A ";
        }
        writeln!(writer, "{{{}}} {}", frame_string, frame_count).unwrap();
    }
    println!("Result saved to {}", output_path.display());
}