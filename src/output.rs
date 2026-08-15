use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::commands::{get_color_change_commands, get_move_commands};
use crate::constants::{C_A, C_D, C_L, C_L2, C_R, C_R2, C_U, TARGET_HEIGHT, TARGET_WIDTH};
use crate::striping::Stripe;
use crate::color::get_color_dist;

// Converts a byte command mask into a string representation mapping to controller buttons.
fn format_cmd(cmd: u8) -> String {
    // Return an empty block string if no command is specified.
    if cmd == 0 {
        return "{}".to_string();
    }
    let mut buttons = Vec::new();
    
    // Evaluate individual bits to append corresponding button labels to the collection.
    if (cmd & C_A) != 0 {
        buttons.push("A");
    }
    if (cmd & C_L) != 0 {
        buttons.push("L");
    }
    if (cmd & C_R) != 0 {
        buttons.push("R");
    }
    if (cmd & C_U) != 0 {
        buttons.push("U");
    }
    if (cmd & C_D) != 0 {
        buttons.push("D");
    }
    if (cmd & C_L2) != 0 {
        buttons.push("L2");
    }
    if (cmd & C_R2) != 0 {
        buttons.push("R2");
    }

    format!("{{{}}}", buttons.join(" "))
}

// Generates instructions for drawing stripes.
pub fn generate_striping_commands<P: AsRef<Path>>(
    filename: P,
    stripes: &[Stripe],
    mut cursor_x: usize,
    mut cursor_y: usize,
    mut current_color: u8,
    cmd_buffer: &mut Vec<u8>,
) -> io::Result<usize> {
    // Initialize the output file and a counter for the total elapsed frames.
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    let mut frames_tracked = 0;

    let max_x = (TARGET_WIDTH - 1) as i32;
    let max_y = (TARGET_HEIGHT - 1) as i32;

    // Increase the pen size
    for _ in 0..3 {
        writeln!(writer, "{{R1}} 2")?;
        writeln!(writer, "{{}} 2")?;
        frames_tracked += 4;
    }

    // Increasing the pen size pushes the cursor off the top/left
    if cursor_x == 0 {
        cursor_x = 1;
    }
    if cursor_y == 0 {
        cursor_y = 1;
    }

    // Process each stripe.
    for stripe in stripes {
        let tx = stripe.start_cursor.0;
        let ty = stripe.start_cursor.1;

        // Calculate the optimal horizontal traversal method by comparing dpad vs stick movement.
        let cost_x_normal = (cursor_x as i32 - tx as i32).abs() * 4;
        let frames_left = cursor_x as i32 / 2 + 6;
        let cost_x_left = frames_left + (tx as i32 * 4);
        let frames_right = (max_x - cursor_x as i32) / 2 + 6;
        let cost_x_right = frames_right + ((max_x - tx as i32) * 4);

        if cost_x_left < cost_x_normal && cost_x_left <= cost_x_right && frames_left > 0 {
            writeln!(writer, "{{}} (0 128) {}", frames_left)?;
            frames_tracked += frames_left as usize;
            cursor_x = 1;
        } else if cost_x_right < cost_x_normal && frames_right > 0 {
            writeln!(writer, "{{}} (255 128) {}", frames_right)?;
            frames_tracked += frames_right as usize;
            cursor_x = max_x as usize;
        }

        // Calculate the optimal vertical traversal method.
        let cost_y_normal = (cursor_y as i32 - ty as i32).abs() * 2;
        let frames_up = cursor_y as i32 / 2 + 6;
        let cost_y_up = frames_up + (ty as i32 * 2);
        let frames_down = (max_y - cursor_y as i32) / 2 + 6;
        let cost_y_down = frames_down + ((max_y - ty as i32) * 2);

        if cost_y_up < cost_y_normal && cost_y_up <= cost_y_down && frames_up > 0 {
            writeln!(writer, "{{}} (128 0) {}", frames_up)?;
            frames_tracked += frames_up as usize;
            cursor_y = 1;
        } else if cost_y_down < cost_y_normal && frames_down > 0 {
            writeln!(writer, "{{}} (128 255) {}", frames_down)?;
            frames_tracked += frames_down as usize;
            cursor_y = max_y as usize;
        }

        // Resolve required d-pad commands to traverse the remaining distance to the start of the stripe and switch to the target color.
        let dx = tx as i16 - cursor_x as i16;
        let dy = ty as i16 - cursor_y as i16;
        let c_delta = get_color_dist(current_color, stripe.color);

        get_move_commands(dx, dy, 0, cmd_buffer);
        get_color_change_commands(c_delta, 0, cmd_buffer);

        // Write the generated point-to-point movement and color commands to the file.
        for &cmd in cmd_buffer.iter() {
            writeln!(writer, "{} 2", format_cmd(cmd))?;
            frames_tracked += 2;
        }

        // Determine analog stick angles and frame durations based on stripe orientation and direction.
        let (stick_str, stripe_frames) = if stripe.is_vertical {
            if stripe.start_cursor.1 < stripe.end_cursor.1 {
                ("(128 255) 90", 90)
            } else {
                ("(128 0) 90", 90)
            }
        } else {
            if stripe.start_cursor.0 < stripe.end_cursor.0 {
                ("(255 128) 160", 160)
            } else {
                ("(0 128) 160", 160)
            }
        };

        // Write the execution commands for the current stripe and tracks the elapsed frames.
        writeln!(writer, "{{A}} 2")?;
        writeln!(writer, "{{A}} {}", stick_str)?;
        frames_tracked += stripe_frames + 2;

        cursor_x = stripe.end_cursor.0.max(1);
        cursor_y = stripe.end_cursor.1.max(1);
        current_color = stripe.color;
    }

    // Return to small cursor
    for _ in 0..3 {
        writeln!(writer, "{{L1}} 2")?;
        writeln!(writer, "{{}} 2")?;
        frames_tracked += 4;
    }

    Ok(frames_tracked)
}

// Appends a sequence of formatted pixel commands to the designated file.
pub fn append_pixel_commands<P: AsRef<Path>>(filename: P, commands: &[u8]) -> io::Result<usize> {
    let file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(filename)?;
    let mut writer = BufWriter::new(file);

    for &cmd in commands {
        writeln!(writer, "{} 2", format_cmd(cmd))?;
    }
    Ok(commands.len() * 2)
}