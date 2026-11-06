//! Memory-mapped i32 weight loader.
//!
//! Weights are stored on disk as little-endian i32 (4 bytes). Instead of reading
//! and converting the whole file, this maps it and converts ranges to canonical
//! Goldilocks on demand, so a whole weight file never has to be resident (or
//! converted) at once. Combined with streaming this keeps peak memory near the
//! working set.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::common::field::Goldilocks;
use crate::common::fixed_point::from_i32;

pub struct WeightMmap {
    mmap: Mmap,
    len: usize,
}

impl WeightMmap {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let len = mmap.len() / 4;
        Ok(Self { mmap, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Convert `[start, start+out.len())` into `out` without an intermediate
    /// allocation. Returns the number of values written (the range is clipped
    /// to the mapped tensor, so a short tail leaves the rest untouched).
    pub fn read_goldilocks_into(&self, start: usize, out: &mut [Goldilocks]) -> usize {
        let end = start.saturating_add(out.len()).min(self.len);
        let n = end.saturating_sub(start);
        for j in 0..n {
            let off = (start + j) * 4;
            let v = i32::from_le_bytes([
                self.mmap[off],
                self.mmap[off + 1],
                self.mmap[off + 2],
                self.mmap[off + 3],
            ]);
            out[j] = from_i32(v);
        }
        n
    }

    /// Convert a contiguous `[start, start+len)` range of i32 to canonical
    /// Goldilocks. Only this range is materialized.
    pub fn read_goldilocks(&self, start: usize, len: usize) -> Vec<Goldilocks> {
        let end = (start + len).min(self.len);
        let mut out = Vec::with_capacity(end.saturating_sub(start));
        for i in start..end {
            let off = i * 4;
            let v = i32::from_le_bytes([
                self.mmap[off],
                self.mmap[off + 1],
                self.mmap[off + 2],
                self.mmap[off + 3],
            ]);
            out.push(from_i32(v));
        }
        out
    }

    pub fn read_all(&self) -> Vec<Goldilocks> {
        self.read_goldilocks(0, self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::field::PrimeCharacteristicRing;

    #[test]
    fn read_goldilocks_into_matches_window_read() {
        let path = std::env::temp_dir().join(format!(
            "zkie-weights-io-{}-{}.bin", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let vals: Vec<i32> = (0..100i32).map(|i| i * 3 - 7).collect();
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        let m = WeightMmap::open(&path).unwrap();
        let want = m.read_goldilocks(10, 30);
        let mut out = vec![Goldilocks::ZERO; 30];
        let n = m.read_goldilocks_into(10, &mut out);
        assert_eq!(n, 30);
        assert_eq!(out.as_slice(), want.as_slice());
        let mut tail = vec![Goldilocks::ZERO; 10];
        assert_eq!(m.read_goldilocks_into(95, &mut tail), 5);
        assert_eq!(&tail[..5], m.read_goldilocks(95, 5).as_slice());
        std::fs::remove_file(&path).unwrap();
    }
}
