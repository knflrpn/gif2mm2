use crate::constants::{C_A, C_D, C_L, C_L2, C_R, C_R2, C_U};

// Overlay a sequence of button presses onto a command sequence to change the palette color.
pub fn get_color_change_commands(color_delta: i8, prev_cmd: u8, commands: &mut Vec<u8>) {
    if color_delta == 0 {
        return;
    }

    // Check if a color change button was already held down.
    let mut add_blank = prev_cmd & (C_R2 | C_L2) != 0;
    let mut need = color_delta;
    let mut idx = 0;

    while need != 0 {
        if add_blank {
            if idx >= commands.len() {
                commands.push(0);
            }
            idx += 1;
        }
        // Blanks are required from here on out to register repeated presses.
        add_blank = true;

        let cmd = if need > 0 {
            need -= 1;
            C_R2
        } else {
            need += 1;
            C_L2
        };

        if idx < commands.len() {
            commands[idx] |= cmd;
        } else {
            commands.push(cmd);
        }
        idx += 1;
    }
}

// Generate a sequence of move commands to reach the target pixel offset,
// considering the previous command to avoid illegal consecutive moves.
pub fn get_move_commands(mut d_x: i16, mut d_y: i16, prev_cmd: u8, commands: &mut Vec<u8>) {
    commands.clear();
    let mut last_cmd = prev_cmd;

    while d_x != 0 || d_y != 0 {
        let (mut potential_cmd_x, mut potential_cmd_y) = (0, 0);
        let mut potential_x = 0;
        let mut potential_y = 0;
        let mut progress = false;

        // First determine which directional moves are both needed and permitted.
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

        // Choose a move, prioritizing the axis with the largest distance to cover.
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
            // Insert an empty command if no progress is possible on this frame.
            commands.push(0);
            last_cmd = 0;
        }
    }
}

// Generate the full command sequence for a movement and color change into the provided buffer.
pub fn derive_commands(
    pixel_offset_x: i16,
    pixel_offset_y: i16,
    color_delta: i8,
    prev_cmd: u8,
    commands: &mut Vec<u8>,
) {
    get_move_commands(pixel_offset_x, pixel_offset_y, prev_cmd, commands);
    get_color_change_commands(color_delta, prev_cmd, commands);

    // If there is only one command, it includes a color change, and A is already down,
    // A must be released before being re-pressed, otherwise the pixel will remain the old color.
    if commands.len() == 1 && (prev_cmd & C_A != 0) && (commands[0] & (C_L2 | C_R2) != 0) {
        commands.push(0);
    }

    // Add A to the last command, pushing a new command if needed.
    if commands.is_empty() {
        commands.push(C_A);
    } else {
        let final_index = commands.len() - 1;
        commands[final_index] |= C_A;
    }
}
