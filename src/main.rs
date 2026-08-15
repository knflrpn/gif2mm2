// src/main.rs

/* In the Mario Maker 2 comment drawing UI,
   U, D, L, R are used to move the cursor,
   L2 and R2 are used to change the color,
   and A is used to draw a pixel.
   The palette has 17 colors, indexed from 0 to 16. The palette is circular.
   The goal of this program is to take an input image and convert it into a sequence of
   button presses that will reproduce the image in the comment drawing UI.
*/

use std::env;

// Declare the external modules
pub mod constants;
mod commands;
mod color;
mod cache;
mod striping;
mod pixel_drawing;
mod optimizer;
mod output;

// Re-export CostCache so modules can still use `crate::CostCache`
pub use cache::CostCache; 

fn load_target_grid(image_path: &str) -> Vec<u8> {
    use constants::{TARGET_HEIGHT, TARGET_WIDTH};

    let img = match image::open(image_path) {
        Ok(img) => img.to_rgb8(),
        Err(e) => {
            eprintln!("Error loading image '{}': {}", image_path, e);
            std::process::exit(1);
        }
    };

    let (width, height) = img.dimensions();
    let mut grid = vec![16u8; (TARGET_WIDTH * TARGET_HEIGHT) as usize];

    for dy in 0..TARGET_HEIGHT as u32 {
        for dx in 0..TARGET_WIDTH as u32 {
            let sx = dx as i32 - (TARGET_WIDTH as i32 / 2) + (width as i32 / 2);
            let sy = dy as i32 - (TARGET_HEIGHT as i32 / 2) + (height as i32 / 2);

            if sx >= 0 && sx < width as i32 && sy >= 0 && sy < height as i32 {
                let pixel = img.get_pixel(sx as u32, sy as u32);
                let color_idx =
                    color::find_closest_color((pixel[0], pixel[1], pixel[2]), color::PALETTE);
                grid[(dy * TARGET_WIDTH as u32 + dx) as usize] = color_idx;
            }
        }
    }
    grid
}

fn evaluate_striping_strategies(
    grid: &[u8],
    cmd_buffer: &mut Vec<u8>,
) -> (Vec<striping::Stripe>, Vec<u8>, (usize, usize), &'static str) {
    use constants::{TARGET_HEIGHT, TARGET_WIDTH};

    let canvas_blank = vec![16u8; (TARGET_WIDTH * TARGET_HEIGHT) as usize];

    let cost_none = striping::estimate_plan_cost(&[], &canvas_blank, grid, cmd_buffer);

    let mut canvas_v = canvas_blank.clone();
    let (stripes_v, cursor_v) = striping::plan_vertical_stripes(grid, &mut canvas_v, (1, 1));
    let cost_v = striping::estimate_plan_cost(&stripes_v, &canvas_v, grid, cmd_buffer);

    let mut canvas_h = canvas_blank.clone();
    let (stripes_h, cursor_h) = striping::plan_horizontal_stripes(grid, &mut canvas_h, (1, 1));
    let cost_h = striping::estimate_plan_cost(&stripes_h, &canvas_h, grid, cmd_buffer);

    let mut canvas_vh = canvas_blank.clone();
    let (mut stripes_vh, cursor_vh_temp) =
        striping::plan_vertical_stripes(grid, &mut canvas_vh, (1, 1));
    let (stripes_vh_h, cursor_vh) =
        striping::plan_horizontal_stripes(grid, &mut canvas_vh, cursor_vh_temp);
    stripes_vh.extend(stripes_vh_h);
    let cost_vh = striping::estimate_plan_cost(&stripes_vh, &canvas_vh, grid, cmd_buffer);

    let mut canvas_hv = canvas_blank.clone();
    let (mut stripes_hv, cursor_hv_temp) =
        striping::plan_horizontal_stripes(grid, &mut canvas_hv, (1, 1));
    let (stripes_hv_v, cursor_hv) =
        striping::plan_vertical_stripes(grid, &mut canvas_hv, cursor_hv_temp);
    stripes_hv.extend(stripes_hv_v);
    let cost_hv = striping::estimate_plan_cost(&stripes_hv, &canvas_hv, grid, cmd_buffer);

    println!("--- Estimated Frame Costs ---");
    println!("No Striping:  {}", cost_none);
    println!("V Only:       {}", cost_v);
    println!("H Only:       {}", cost_h);
    println!("V then H:     {}", cost_vh);
    println!("H then V:     {}", cost_hv);

    let min_cost = *[cost_none, cost_v, cost_h, cost_vh, cost_hv]
        .iter()
        .min()
        .unwrap();

    if min_cost == cost_none {
        (vec![], canvas_blank, (0, 0), "No Striping")
    } else if min_cost == cost_v {
        (stripes_v, canvas_v, cursor_v, "Vertical Only")
    } else if min_cost == cost_h {
        (stripes_h, canvas_h, cursor_h, "Horizontal Only")
    } else if min_cost == cost_vh {
        (stripes_vh, canvas_vh, cursor_vh, "Vertical then Horizontal")
    } else {
        (stripes_hv, canvas_hv, cursor_hv, "Horizontal then Vertical")
    }
}

fn run_pixel_drawing_sweep(
    grid: &[u8],
    base_canvas: &[u8],
    start_cx: usize,
    start_cy: usize,
    start_color: u8,
    cmd_buffer: &mut Vec<u8>,
) -> (Vec<u8>, pixel_drawing::SearchParams) {
    println!("\nStarting parameter sweep for pixel drawing...");

    let weights: [f32; 2] = [0.2, 1.8];
    let biases: [f32; 9] = [-1.5, -1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0, 1.5];

    let mut best_path = Vec::new();
    let mut best_params = pixel_drawing::SearchParams {
        weight_x: 1.0,
        weight_y: 1.0,
        bias_x: 0.0,
        bias_y: 0.0,
    };
    let mut min_cost = usize::MAX;

    let mut cost_cache = CostCache::new();
    let mut test_canvas = vec![16u8; base_canvas.len()];

    let total_sweeps = weights.len() * weights.len() * biases.len() * biases.len();
    let mut current_sweep = 0;
    let mut current_progress_step = 10;

    for &wx in &weights {
        for &wy in &weights {
            for &bx in &biases {
                for &by in &biases {
                    let params = pixel_drawing::SearchParams {
                        weight_x: wx,
                        weight_y: wy,
                        bias_x: bx,
                        bias_y: by,
                    };

                    test_canvas.copy_from_slice(base_canvas);

                    let (path, cost) = pixel_drawing::plan_pixel_path(
                        grid,
                        &mut test_canvas,
                        start_cx,
                        start_cy,
                        start_color,
                        &params,
                        &mut cost_cache,
                        cmd_buffer,
                    );

                    if cost < min_cost {
                        min_cost = cost;
                        best_path = path;
                        best_params = params;
                    }

                    // Progress indicator
                    current_sweep += 1;
                    if (current_sweep * 100) / total_sweeps >= current_progress_step {
                        println!("  {}%", current_progress_step);
                        current_progress_step += 10;
                    }
                }
            }
        }
    }

    /* Optimization phase on final visit order */

    println!("Running re-ordering optimization...");
    optimizer::minor_optimize_visit_order(&mut best_path);

    // Multiple fast passes
    println!("Running rapid sliding window optimization...");
    for _ in 0..5 {
        optimizer::major_optimize_visit_order(
            &mut best_path,
            7,
            grid,
            &mut cost_cache,
            cmd_buffer,
            false,
        );
    }
    println!("Running re-ordering optimization...");
    optimizer::minor_optimize_visit_order(&mut best_path);

    // One slow pass
    println!("Running long sliding window optimization...");
    optimizer::major_optimize_visit_order(
        &mut best_path,
        12,
        grid,
        &mut cost_cache,
        cmd_buffer,
        true,
    );
    println!("Running re-ordering optimization...");
    optimizer::minor_optimize_visit_order(&mut best_path);

    /* Generate actual command sequence */
    let best_pixel_cmds = pixel_drawing::generate_commands_from_path(
        &best_path,
        grid,
        start_cx,
        start_cy,
        start_color,
        cmd_buffer,
    );

    (best_pixel_cmds, best_params)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_image>", args[0]);
        std::process::exit(1);
    }
    let image_path = &args[1];

    let mut cmd_buffer = Vec::with_capacity(64);

    let grid = load_target_grid(image_path);

    let (best_stripes, best_canvas, end_cursor, best_name) =
        evaluate_striping_strategies(&grid, &mut cmd_buffer);

    println!("\nSelected Strategy: {}", best_name);
    println!("Total stripes planned: {}", best_stripes.len());
    println!("Cursor left at ({},{})", end_cursor.0, end_cursor.1);

    let path = std::path::Path::new(image_path);
    let temp_output_file = path.with_extension("tmp.txt");
    let mut actual_total_frames = 0;

    if !best_stripes.is_empty() {
        match output::generate_striping_commands(
            &temp_output_file,
            &best_stripes,
            0,
            0,
            0,
            &mut cmd_buffer,
        ) {
            Ok(frames) => {
                actual_total_frames += frames;
                println!("Successfully wrote striping commands.");
            }
            Err(e) => eprintln!("Failed to write striping commands: {}", e),
        };
    } else {
        let _ = std::fs::File::create(&temp_output_file);
    }

    let start_color = best_stripes.last().map(|s| s.color).unwrap_or(0);
    let (best_pixel_cmds, best_params) = run_pixel_drawing_sweep(
        &grid,
        &best_canvas,
        end_cursor.0,
        end_cursor.1,
        start_color,
        &mut cmd_buffer,
    );

    println!("Best parameters found: {:?}", best_params);

    match output::append_pixel_commands(&temp_output_file, &best_pixel_cmds) {
        Ok(frames) => {
            actual_total_frames += frames;
            println!("Appended {} pixel commands.", best_pixel_cmds.len());
        }
        Err(e) => eprintln!("Failed to write pixel commands: {}", e),
    }

    let total_seconds = actual_total_frames / 60;
    let mins = total_seconds / 60;
    let secs = total_seconds % 60;

    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new(""));
    let final_filename = format!("{}_{:02}m_{:02}s.txt", stem, mins, secs);
    let final_path = parent.join(final_filename);

    match std::fs::rename(&temp_output_file, &final_path) {
        Ok(_) => println!("Successfully saved script to {:?}", final_path),
        Err(e) => eprintln!("Failed to rename final file to {:?}: {}", final_path, e),
    }

    println!(
        "Duration: {} frames ({:02}m {:02}s)",
        actual_total_frames, mins, secs
    );
}