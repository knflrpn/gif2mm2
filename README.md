# gif2mm2

gif2mm2 is a command-line tool that converts a gif to a sequence of button presses to draw the image in the Super Mario Maker 2 comment system. Users can then send the commands through a [SwiCC](https://github.com/knflrpn/SwiCC_RP2040).

## Features

- Converts gif to a sequence of button presses
- Optimizes the path to reduce the time required for drawing
- Supports both filled and unfilled background

## Usage

Usage: gif2mm2 <input_file> <frames_per_command> (optional, default 2).

The `input_file` should be a GIF image with dimensions of 320x180.

It is recommended to leave `frames_per_command` as default unless you have a specific reason to use it.

## Implementation Details

gif2mm2 uses a greedy algorithm to find a decent path for drawing the image.  After the initial path is derived, it attempts to optimize based on some of the known weaknessess of greedy algorithms.  The optimized path is then converted into a command sequence that can be sent to a SwiCC.

## Contributing

If you are good at algorithms and want to create a more optimal solution to this traveling salesman problem, get in touch.

## License

This project is released under the MIT License.
