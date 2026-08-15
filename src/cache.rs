// src/cache.rs

pub struct CostCache {
    // Stores both cost (top 24 bits) and last command (bottom 8 bits)
    table: Vec<u32>,
}

impl CostCache {
    pub fn new() -> Self {
        Self {
            table: vec![u32::MAX; 64 * 64 * 17 * 128],
        }
    }

    #[inline(always)]
    fn get_index(dx: i16, dy: i16, color_delta: i8, prev_cmd: u8) -> Option<usize> {
        let ux = (dx + 31) as u16;
        let uy = (dy + 31) as u16;

        if ux > 62 || uy > 62 {
            return None;
        }

        let uc = (color_delta + 8) as usize;
        let cmd = prev_cmd as usize;

        Some(((((ux as usize) << 6) + (uy as usize)) * 17 + uc) << 7 | cmd)
    }

    pub fn get_or_insert_with<F: FnOnce(&mut Vec<u8>)>(
        &mut self,
        dx: i16,
        dy: i16,
        color_delta: i8,
        prev_cmd: u8,
        buffer: &mut Vec<u8>,
        compute: F,
    ) -> (usize, u8) {
        if let Some(idx) = Self::get_index(dx, dy, color_delta, prev_cmd) {
            let val = self.table[idx];
            if val != u32::MAX {
                ((val >> 8) as usize, (val & 0xFF) as u8)
            } else {
                compute(buffer);
                let cost = buffer.len();
                let last_cmd = buffer.last().copied().unwrap_or(0);
                self.table[idx] = ((cost as u32) << 8) | (last_cmd as u32);
                (cost, last_cmd)
            }
        } else {
            compute(buffer);
            (buffer.len(), buffer.last().copied().unwrap_or(0))
        }
    }

    /// Fast inline lookup to avoid passing closures/buffers when a cost is already cached.
    #[inline(always)]
    pub fn get(&self, dx: i16, dy: i16, color_delta: i8, prev_cmd: u8) -> Option<(usize, u8)> {
        if let Some(idx) = Self::get_index(dx, dy, color_delta, prev_cmd) {
            let val = self.table[idx];
            if val != u32::MAX {
                return Some(((val >> 8) as usize, (val & 0xFF) as u8));
            }
        }
        None
    }
}
