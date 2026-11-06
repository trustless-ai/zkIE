//! O4 measurement: verifier allocation/time for the public broadcast selector.
//!
//! Before O4 the batched verifier materialized eq_b[pos*K+d] = eq_u[pos]
//! (S*K elements) and evaluated its MLE. After O4 it evaluates eq_u directly
//! at the high challenge variables ch[logK..]. This example compares the
//! pre-O4 reference verifier with the current one on the same honest proof.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use zkie_core::common::batched_matmul::{forward, prove, verify, BatchedMatmulProof};
use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::mle;
use zkie_core::common::sumcheck::{verify_virtual, virtual_eval, VirtualProof};
use zkie_core::common::transcript::Transcript;

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

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn verifier_transcript(s: usize, n: usize) -> Transcript {
    let mut t = Transcript::new(b"o4-verify");
    t.absorb_u64(0x04A4);
    // Match the prover's transcript state: u and v were squeezed from it.
    for _ in 0..s.trailing_zeros() {
        t.field();
    }
    for _ in 0..n.trailing_zeros() {
        t.field();
    }
    t
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn terms() -> Vec<(Goldilocks, Vec<usize>)> {
    vec![(Goldilocks::ONE, vec![0usize, 1, 2])]
}

/// Pre-O4 verifier: materialize the S*K broadcast column and evaluate it.
fn reference_verify(
    proof: &BatchedMatmulProof,
    s: usize,
    k: usize,
    u: &[Goldilocks],
    transcript: &mut Transcript,
) -> (bool, Vec<Goldilocks>) {
    let eq_u = mle::eq_evals(u);
    let mut eq_b = vec![Goldilocks::ZERO; s * k];
    for pos in 0..s {
        for d in 0..k {
            eq_b[pos * k + d] = eq_u[pos];
        }
    }
    let (ok, ch, final_claim) = verify_virtual(
        &proof.sumcheck,
        &terms(),
        proof.claimed,
        (s * k).trailing_zeros() as usize,
        transcript,
    );
    if !ok {
        return (false, ch);
    }
    let eq_final = mle::eval(&eq_b, &ch);
    let final_evals = vec![
        proof.sumcheck.final_evals[0],
        proof.sumcheck.final_evals[1],
        eq_final,
    ];
    (final_claim == virtual_eval(&terms(), &final_evals), ch)
}

fn main() {
    let (s, k, n) = (512usize, 2048usize, 8usize);
    let mut rng = XorShift64::new(0x04A4);
    let seg: Vec<Goldilocks> = (0..s * k).map(|_| rng.field()).collect();
    let w: Vec<Goldilocks> = (0..s * k * n).map(|_| rng.field()).collect();
    let g = forward(&seg, &w, s, k, n);

    let mut tp = Transcript::new(b"o4-verify");
    tp.absorb_u64(0x04A4);
    let u: Vec<Goldilocks> = (0..s.trailing_zeros() as usize).map(|_| tp.field()).collect();
    let v: Vec<Goldilocks> = (0..n.trailing_zeros() as usize).map(|_| tp.field()).collect();
    let (proof, _ch) = prove(&seg, &w, &g, s, k, n, &u, &v, &mut tp);

    // Warm both verifiers.
    {
        let mut t = verifier_transcript(s, n);
        let _ = verify(&proof, s, k, &u, &mut t);
        let mut t = verifier_transcript(s, n);
        let _ = reference_verify(&proof, s, k, &u, &mut t);
    }

    let (a0, b0) = snapshot();
    let t0 = std::time::Instant::now();
    let mut tv = verifier_transcript(s, n);
    let (ok_opt, ch_opt) = verify(&proof, s, k, &u, &mut tv);
    let opt_ms = t0.elapsed().as_secs_f64() * 1e3;
    let (a1, b1) = snapshot();

    let (a2, b2) = snapshot();
    let t1 = std::time::Instant::now();
    let mut tv = verifier_transcript(s, n);
    let (ok_ref, ch_ref) = reference_verify(&proof, s, k, &u, &mut tv);
    let ref_ms = t1.elapsed().as_secs_f64() * 1e3;
    let (a3, b3) = snapshot();

    println!(
        "O4 s={} k={} n={} optimized_ok={} optimized_allocs={} optimized_bytes={} optimized_ms={:.3}",
        s, k, n, ok_opt, a1 - a0, b1 - b0, opt_ms
    );
    println!(
        "O4 s={} k={} n={} reference_ok={} reference_allocs={} reference_bytes={} reference_ms={:.3}",
        s, k, n, ok_ref, a3 - a2, b3 - b2, ref_ms
    );
    let proof_bytes = bincode::serialize(&proof).expect("serialize O4 proof");
    println!(
        "O4 same_result={} same_challenges={} proof_len={} proof_hash={:016x} hwm_kb={}",
        ok_opt == ok_ref,
        ch_opt == ch_ref,
        proof_bytes.len(),
        fnv(&proof_bytes),
        rss_kb()
    );
    let _: VirtualProof = proof.sumcheck;
}
