//! O3 measurement: allocations and time for the virtual sumcheck hot loop.
//!
//! Compares the pre-O3 map+reduce implementation (one Vec per element pair)
//! with the current fold+reduce implementation (one reused accumulator per
//! Rayon worker). The proof bytes must be identical.
//!
//! Run with `RAYON_NUM_THREADS=1` and with the default pool to check
//! single-thread/multi-thread equality of the proof output.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::common::sumcheck::{prove_virtual, VirtualProof};
use zkie_core::common::transcript::Transcript;

use rayon::prelude::*;

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn snapshot() -> (usize, usize) {
    (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed))
}

fn status_kb(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn rss_kb() -> u64 { status_kb("VmHWM:") }
fn rss_now_kb() -> u64 { status_kb("VmRSS:") }

fn clock_ticks() -> f64 {
    std::process::Command::new("getconf")
        .arg("CLK_TCK")
        .output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(100.0)
}

/// Process CPU time (user+system) in milliseconds, from /proc on Linux.
fn cpu_ms() -> f64 {
    let tck = clock_ticks();
    let Ok(s) = std::fs::read_to_string("/proc/self/stat") else { return 0.0; };
    let t = s.replace(')', " ");
    let f: Vec<&str> = t.split_whitespace().collect();
    let utime: f64 = f.get(13).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let stime: f64 = f.get(14).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    (utime + stime) * 1000.0 / tck
}

/// Pre-O3 implementation: `map` allocates one Vec per element pair.
fn reference_virtual(
    mles: &[&[Goldilocks]],
    terms: &[(Goldilocks, Vec<usize>)],
    claimed: Goldilocks,
    transcript: &mut Transcript,
) -> (VirtualProof, Vec<Goldilocks>) {
    let n = mles[0].len();
    let t = n.trailing_zeros() as usize;
    let max_deg = terms.iter().map(|(_, idxs)| idxs.len()).max().unwrap_or(0);
    transcript.absorb_field(claimed);
    let mut bufs: Vec<Vec<Goldilocks>> = mles.iter().map(|m| m.to_vec()).collect();
    let mut rounds = Vec::with_capacity(t);
    let mut challenges = Vec::with_capacity(t);
    for _ in 0..t {
        let half = bufs[0].len() / 2;
        let pvals: Vec<Goldilocks> = (0..half)
            .into_par_iter()
            .map(|s| {
                let mut p = vec![Goldilocks::ZERO; max_deg + 1];
                for (coeff, idxs) in terms {
                    for k in 0..=max_deg {
                        let mut prod = *coeff;
                        for &j in idxs {
                            let f0 = bufs[j][2 * s];
                            let f1 = bufs[j][2 * s + 1];
                            let fk = from_i64(k as i64) * f1 - from_i64(k as i64 - 1) * f0;
                            prod = prod * fk;
                        }
                        p[k] = p[k] + prod;
                    }
                }
                p
            })
            .reduce(
                || vec![Goldilocks::ZERO; max_deg + 1],
                |mut a, b| {
                    for i in 0..=max_deg {
                        a[i] = a[i] + b[i];
                    }
                    a
                },
            );
        for &p in &pvals {
            transcript.absorb_field(p);
        }
        rounds.push(pvals);
        let r = transcript.field();
        challenges.push(r);
        bufs.par_iter_mut().for_each(|buf| {
            for s in 0..half {
                let f0 = buf[2 * s];
                let f1 = buf[2 * s + 1];
                buf[s] = (Goldilocks::ONE - r) * f0 + r * f1;
            }
            buf.truncate(half);
        });
    }
    let final_evals = bufs.iter().map(|b| b[0]).collect();
    (VirtualProof { rounds, final_evals }, challenges)
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn main() {
    let n = 1usize << 18;
    let mut rng = XorShift64::new(0x03A11);
    let f: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
    let g: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
    let h: Vec<Goldilocks> = (0..n).map(|_| rng.field()).collect();
    let terms: Vec<(Goldilocks, Vec<usize>)> = vec![
        (Goldilocks::from_u64(3), vec![0, 1]),
        (from_i64(-2), vec![1, 2]),
        (Goldilocks::ONE, vec![0, 1, 2]),
    ];
    let claimed: Goldilocks = (0..n)
        .map(|i| {
            Goldilocks::from_u64(3) * f[i] * g[i]
                + from_i64(-2) * g[i] * h[i]
                + f[i] * g[i] * h[i]
        })
        .sum();
    let mles: Vec<&[Goldilocks]> = vec![&f, &g, &h];

    // Warm the Rayon pool and the allocator before measuring.
    {
        let mut t = Transcript::new(b"o3-warmup");
        let _ = prove_virtual(&mles, &terms, claimed, &mut t);
        let mut t = Transcript::new(b"o3-warmup");
        let _ = reference_virtual(&mles, &terms, claimed, &mut t);
    }

    let rss0 = rss_now_kb();
    let cpu0 = cpu_ms();
    let (a0, b0) = snapshot();
    let t0 = std::time::Instant::now();
    let mut ts = Transcript::new(b"o3-measure");
    let (opt, opt_ch) = prove_virtual(&mles, &terms, claimed, &mut ts);
    let (a1, b1) = snapshot();
    let opt_ms = t0.elapsed().as_secs_f64() * 1e3;
    let opt_cpu_ms = cpu_ms() - cpu0;
    let rss1 = rss_now_kb();

    let rss2 = rss_now_kb();
    let cpu1 = cpu_ms();
    let (a2, b2) = snapshot();
    let t1 = std::time::Instant::now();
    let mut ts = Transcript::new(b"o3-measure");
    let (rf, rf_ch) = reference_virtual(&mles, &terms, claimed, &mut ts);
    let (a3, b3) = snapshot();
    let rf_ms = t1.elapsed().as_secs_f64() * 1e3;
    let rf_cpu_ms = cpu_ms() - cpu1;
    let rss3 = rss_now_kb();

    println!(
        "O3 n={} terms={} threads={} optimized_allocs={} optimized_bytes={} optimized_ms={:.3} optimized_cpu_ms={:.3} rss_before_kb={} rss_after_kb={}",
        n,
        terms.len(),
        rayon::current_num_threads(),
        a1 - a0,
        b1 - b0,
        opt_ms,
        opt_cpu_ms,
        rss0,
        rss1
    );
    println!(
        "O3 n={} terms={} threads={} reference_allocs={} reference_bytes={} reference_ms={:.3} reference_cpu_ms={:.3} rss_before_kb={} rss_after_kb={}",
        n,
        terms.len(),
        rayon::current_num_threads(),
        a3 - a2,
        b3 - b2,
        rf_ms,
        rf_cpu_ms,
        rss2,
        rss3
    );
    let opt_bytes = bincode::serialize(&opt).unwrap();
    let rf_bytes = bincode::serialize(&rf).unwrap();
    println!(
        "O3 proof_equal={} challenges_equal={} proof_len={} proof_hash={:016x} hwm_kb={}",
        opt_bytes == rf_bytes,
        opt_ch == rf_ch,
        opt_bytes.len(),
        fnv(&opt_bytes),
        rss_kb()
    );
}
