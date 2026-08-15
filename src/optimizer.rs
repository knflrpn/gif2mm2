// Represent a single point containing coordinates and color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Node {
    pub x: u16,
    pub y: u16,
    pub color: u8,
}

impl Node {
    // Instantiate a Node by calculating the array index from coordinates.
    #[inline(always)]
    pub fn from_coord(coord: (usize, usize), target: &[u8]) -> Self {
        let x = coord.0 as u16;
        let y = coord.1 as u16;
        let color = target[(y as usize) * crate::constants::TARGET_WIDTH + (x as usize)];
        Self { x, y, color }
    }

    // Converts the internal coordinates back into a standard tuple.
    #[inline(always)]
    pub fn to_coord(self) -> (usize, usize) {
        (self.x as usize, self.y as usize)
    }
}

/// Persistent allocations reused across sliding windows to eliminate heap malloc/free.
pub struct OptimizerBuffers {
    pub middle_nodes: Vec<Node>,
    pub current_permutation: Vec<Node>,
    pub optimal_permutation: Vec<Node>,
}

impl OptimizerBuffers {
    // Allocate vectors with a predefined capacity for sliding window operations.
    pub fn new() -> Self {
        Self {
            middle_nodes: Vec::with_capacity(16),
            current_permutation: Vec::with_capacity(16),
            optimal_permutation: Vec::with_capacity(16),
        }
    }
}

// Relocates small disconnected blocks of coordinates to closer segments in the path.
pub fn rearrange_pixels(
    visit_order: &mut Vec<(usize, usize)>,
    isolated_block_size: usize,
    isolation_distance: usize,
) {
    let mut isolated_blocks: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut i = 0;
    
    // Scan the coordinate order to group sequential points.
    while i < visit_order.len() {
        let start_index = i;
        let mut end_index = i;
        let (mut prev_x, mut prev_y) = visit_order[i];

        // Continue grouping as long as points fall within the allowed isolation distance.
        while end_index < visit_order.len() - 1 {
            let (next_x, next_y) = visit_order[end_index + 1];
            let x_dist = next_x.abs_diff(prev_x);
            let y_dist = next_y.abs_diff(prev_y);
            
            if x_dist > isolation_distance || y_dist > isolation_distance {
                break;
            }
            
            prev_x = next_x;
            prev_y = next_y;
            end_index += 1;
        }

        let block_size = end_index - start_index + 1;
        
        // Extract the block if it meets the size criteria for isolated blocks.
        if block_size <= isolated_block_size {
            let block = visit_order.drain(start_index..=end_index).collect();
            isolated_blocks.push(block);
            i = start_index;
        } else {
            i = end_index + 1;
        }
    }
    
    // Reinsert the disconnected blocks at the closest possible locations.
    for block in isolated_blocks {
        let nearest_index = find_nearest_index(&visit_order, &block[0]);
        visit_order.splice(nearest_index..nearest_index, block);
    }
}

// Scans the current order to find an insertion index closest to a target coordinate.
pub fn find_nearest_index(visit_order: &[(usize, usize)], coord: &(usize, usize)) -> usize {
    let mut nearest_index = 0;
    let mut min_distance = usize::MAX;
    
    for (j, &(visit_x, visit_y)) in visit_order.iter().enumerate() {
        let distance = coord.0.abs_diff(visit_x).max(coord.1.abs_diff(visit_y));
        
        // Ensure distance improvement while preserving sequence integrity.
        if j < visit_order.len() - 1 {
            let next_coord = visit_order[j + 1];
            let next_distance = next_coord
                .0
                .abs_diff(visit_x)
                .max(next_coord.1.abs_diff(visit_y));
                
            if distance < min_distance && next_distance >= 2 {
                min_distance = distance;
                nearest_index = j;
            }
        }
    }
    nearest_index
}

// Slices the visitation sequence at large distance gaps and reorders the segments.
pub fn optimize_chunks(visit_order: &mut Vec<(usize, usize)>, gap_distance: usize) {
    let mut chunks: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut current_chunk: Vec<(usize, usize)> = Vec::new();
    
    // Break the path into chunks whenever a gap exceeds the specified threshold.
    for i in 0..visit_order.len() {
        current_chunk.push(visit_order[i]);
        if i < visit_order.len() - 1 {
            let curr_pixel = visit_order[i];
            let next_pixel = visit_order[i + 1];
            let distance = curr_pixel
                .0
                .abs_diff(next_pixel.0)
                .max(curr_pixel.1.abs_diff(next_pixel.1));
                
            if distance > gap_distance {
                chunks.push(current_chunk.clone());
                current_chunk.clear();
            }
        }
    }
    
    // Append any residual coordinates.
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    
    if chunks.is_empty() {
        return;
    }

    let mut new_visit_order: Vec<(usize, usize)> = chunks.remove(0);
    
    // Greedily append chunks based on proximity to the end of the newly constructed sequence.
    while !chunks.is_empty() {
        let end_coord = *new_visit_order.last().unwrap();
        let mut min_distance = usize::MAX;
        let mut min_index = 0;
        
        for (i, chunk) in chunks.iter().enumerate() {
            let start_coord = chunk[0];
            let distance = start_coord
                .0
                .abs_diff(end_coord.0)
                .max(start_coord.1.abs_diff(end_coord.1));
                
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

// Recursively tests node sequences to discover the lowest cost path for the segment.
pub fn generate_and_search_permutations(
    current_permutation: &mut Vec<Node>,
    remaining_nodes: &mut Vec<Node>,
    current_cost: usize,
    optimal_cost: &mut usize,
    optimal_permutation: &mut Vec<Node>,
    curr_node: Node,
    end_node: Node,
    cache: &mut crate::CostCache,
    cmd_buffer: &mut Vec<u8>,
    prev_cmd: u8,
) {
    let remaining_len = remaining_nodes.len();

    // Fast heuristic bound check to prune highly inefficient branches early.
    let lower_bound = if remaining_len > 0 {
        let mut min_dist_to_next = usize::MAX;
        
        for rn in remaining_nodes.iter() {
            let dist = (curr_node.x.abs_diff(rn.x).max(curr_node.y.abs_diff(rn.y))) as usize;
            if dist < min_dist_to_next {
                min_dist_to_next = dist;
            }
        }
        min_dist_to_next + (remaining_len - 1)
    } else {
        0
    };

    // Abandon calculation if the lowest possible cost still exceeds the best known cost.
    if current_cost + lower_bound >= *optimal_cost {
        return;
    }

    // Base case: Evaluated full sequence.
    if remaining_len == 0 {
        let dx = end_node.x as i16 - curr_node.x as i16;
        let dy = end_node.y as i16 - curr_node.y as i16;
        let color_delta = crate::color::get_color_dist(curr_node.color, end_node.color);

        // Fetch or calculate cost for transitioning to the end node.
        let last_cost = if let Some((cost, _)) = cache.get(dx, dy, color_delta, prev_cmd) {
            cost
        } else {
            let (cost, _) =
                cache.get_or_insert_with(dx, dy, color_delta, prev_cmd, cmd_buffer, |buf| {
                    crate::commands::derive_commands(dx, dy, color_delta, prev_cmd, buf);
                });
            cost
        };

        let cost = current_cost + last_cost;
        
        // Record sequence if it establishes a new minimum cost.
        if cost < *optimal_cost {
            *optimal_cost = cost;
            optimal_permutation.clear();
            optimal_permutation.extend_from_slice(current_permutation);
        }
        return;
    }

    // Permutation search loop.
    for i in 0..remaining_len {
        // Swap rather than shift to maintain loop performance.
        remaining_nodes.swap(i, remaining_len - 1);
        let next_node = remaining_nodes.pop().unwrap();

        let dx = next_node.x as i16 - curr_node.x as i16;
        let dy = next_node.y as i16 - curr_node.y as i16;
        let color_delta = crate::color::get_color_dist(curr_node.color, next_node.color);

        // Fast path inline lookup to avoid redundant cost derivation.
        let (cmd_cost, new_prev_cmd) = if let Some(res) = cache.get(dx, dy, color_delta, prev_cmd) {
            res
        } else {
            cache.get_or_insert_with(dx, dy, color_delta, prev_cmd, cmd_buffer, |buf| {
                crate::commands::derive_commands(dx, dy, color_delta, prev_cmd, buf);
            })
        };

        let new_cost = current_cost + cmd_cost;
        current_permutation.push(next_node);

        // Continue search with the newly evaluated node.
        generate_and_search_permutations(
            current_permutation,
            remaining_nodes,
            new_cost,
            optimal_cost,
            optimal_permutation,
            next_node, // Pass updated current node as value.
            end_node,
            cache,
            cmd_buffer,
            new_prev_cmd, // Pass updated prev_cmd as value.
        );

        // Revert vector state to continue permutation generation.
        current_permutation.pop();
        remaining_nodes.push(next_node);
        remaining_nodes.swap(i, remaining_len - 1);
    }
}

// Assesses a segment of the path for an optimal ordering
pub fn optimize_segment(
    visit_order: &mut [(usize, usize)],
    index: usize,
    length: usize,
    target: &[u8],
    cache: &mut crate::CostCache,
    cmd_buffer: &mut Vec<u8>,
    buffers: &mut OptimizerBuffers,
) {
    if length <= 3 {
        return;
    }

    let start_node = Node::from_coord(visit_order[index], target);
    let end_node = Node::from_coord(visit_order[index + length - 1], target);

    // Extract central segment nodes for evaluation.
    buffers.middle_nodes.clear();
    for &coord in &visit_order[index + 1..index + length - 1] {
        buffers.middle_nodes.push(Node::from_coord(coord, target));
    }

    // Ascertain previous state to correctly measure transition overhead.
    let prev_prev_cmd = if index < 1 {
        0
    } else {
        let prev_node = Node::from_coord(visit_order[index], target);
        let prev_prev_node = Node::from_coord(visit_order[index - 1], target);
        let dx = prev_node.x as i16 - prev_prev_node.x as i16;
        let dy = prev_node.y as i16 - prev_prev_node.y as i16;
        let color_delta = crate::color::get_color_dist(prev_prev_node.color, prev_node.color);

        let (_, pcmd) = cache.get_or_insert_with(
            dx,
            dy,
            color_delta,
            crate::constants::C_A,
            cmd_buffer,
            |buf| {
                crate::commands::derive_commands(dx, dy, color_delta, crate::constants::C_A, buf);
            },
        );
        pcmd
    };

    // Precalculate baseline cost using the existing unoptimized layout.
    let mut baseline_cost = 0;
    let mut curr_cmd = prev_prev_cmd;
    let mut curr_node = start_node;

    for &next_node in &buffers.middle_nodes {
        let dx = next_node.x as i16 - curr_node.x as i16;
        let dy = next_node.y as i16 - curr_node.y as i16;
        let color_delta = crate::color::get_color_dist(curr_node.color, next_node.color);

        let (cost, next_cmd) =
            cache.get_or_insert_with(dx, dy, color_delta, curr_cmd, cmd_buffer, |buf| {
                crate::commands::derive_commands(dx, dy, color_delta, curr_cmd, buf);
            });
        baseline_cost += cost;
        curr_cmd = next_cmd;
        curr_node = next_node;
    }

    let dx = end_node.x as i16 - curr_node.x as i16;
    let dy = end_node.y as i16 - curr_node.y as i16;
    let color_delta = crate::color::get_color_dist(curr_node.color, end_node.color);

    let (final_cost, _) =
        cache.get_or_insert_with(dx, dy, color_delta, curr_cmd, cmd_buffer, |buf| {
            crate::commands::derive_commands(dx, dy, color_delta, curr_cmd, buf);
        });
    baseline_cost += final_cost;

    let mut optimal_cost = baseline_cost;
    buffers.current_permutation.clear();
    buffers.optimal_permutation.clear();

    // Trigger recursive search for an improved ordering.
    generate_and_search_permutations(
        &mut buffers.current_permutation,
        &mut buffers.middle_nodes,
        0,
        &mut optimal_cost,
        &mut buffers.optimal_permutation,
        start_node,
        end_node,
        cache,
        cmd_buffer,
        prev_prev_cmd,
    );

    // Apply new sequence in-place without slice reallocation/splicing.
    if !buffers.optimal_permutation.is_empty() {
        for (i, node) in buffers.optimal_permutation.iter().enumerate() {
            visit_order[index + 1 + i] = node.to_coord();
        }
    }
}

// Drives the segment optimizer across the entire path using sliding windows.
pub fn major_optimize_visit_order(
    visit_order: &mut Vec<(usize, usize)>,
    optimize: usize,
    target: &[u8],
    cache: &mut crate::CostCache,
    cmd_buffer: &mut Vec<u8>,
    print_progress: bool,
) {
    if visit_order.len() > optimize {
        let total = visit_order.len() - optimize + 1;
        let mut progress_step = 10;
        
        // Instantiated once for all passes to reuse heap allocations.
        let mut buffers = OptimizerBuffers::new(); 

        for (progress, index) in (0..total).enumerate() {
            optimize_segment(
                visit_order,
                index,
                optimize,
                target,
                cache,
                cmd_buffer,
                &mut buffers,
            );

            // Log status to output stream periodically.
            if print_progress && (progress * 100 / total) >= progress_step {
                println!("  {}%", progress_step);
                progress_step += 10;
            }
        }
    }
}

// Executes basic heuristics to improve path before thorough optimization.
pub fn minor_optimize_visit_order(visit_order: &mut Vec<(usize, usize)>) {
    optimize_chunks(visit_order, 30);
    rearrange_pixels(visit_order, 10, 6);
    rearrange_pixels(visit_order, 100, 30);
}