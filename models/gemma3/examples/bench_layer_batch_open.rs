//! Gemma 3 horizontal-batching probe: the real extracted weight tensors of
//! all 18 layers, one claim each, committed and opened with WHIR at 90-bit
//! security.
//!
//! Compares per-tensor commit+open (horizontal batching OFF, one FRI proof per
//! claim) against per-arity batch commit + multi-point batch open (horizontal
//! batching ON, one FRI proof per arity group), exactly the split the
//! committed shard-DAG path performs in `commit_model_weights`, including its
//! PoW-budget group packing (num_vars <= 26).
//!
//! Reports both wall time and the serialized proof size (bincode) for what a
//! verifier must receive in each path.

use std::time::{Duration, Instant};

use zkie_core::common::field::{Goldilocks, XorShift64};
use zkie_core::common::fixed_point::from_i32;
use zkie_core::common::mle;
use zkie_core::pcs::batch_open::{prove_batch_open_committed, verify_batch_open_committed};
use zkie_core::pcs::committed::BatchCtx;
use zkie_core::pcs::whir::Whir;

const SECURITY: usize = 90;

fn load_i32(path: &str) -> Vec<Goldilocks> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| from_i32(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

/// Per-layer tensor files only: `L<digit>..._i32.bin`. Shared tensors
/// (embed/lm_head/rope/tables/inputs) don't carry an `L<digit>` prefix.
fn layer_files(dir: &str) -> Vec<String> {
    let mut files: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| {
            let rest = n.strip_prefix('L').unwrap_or("");
            rest.chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
                && n.ends_with("_i32.bin")
        })
        .collect();
    files.sort();
    files
}

fn mb(n: usize) -> String {
    format!("{:.2} MB", n as f64 / 1e6)
}

fn main() {
    let dir = "models/gemma3/weights";
    let files = layer_files(dir);
    let tensors: Vec<Vec<Goldilocks>> = files.iter().map(|f| load_i32(&format!("{dir}/{f}"))).collect();
    let mut hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for t in &tensors {
        *hist.entry(t.len().trailing_zeros() as usize).or_default() += 1;
    }
    println!(
        "{} layer tensors: {:?}",
        tensors.len(),
        hist.iter().map(|(a, n)| format!("2^{a} x{n}")).collect::<Vec<_>>()
    );

    // One claim per tensor at a random point (the shard-DAG path claims each
    // weight at roughly one point per shard boundary).
    let mut rng = XorShift64::new(0xBEEF);
    let points: Vec<Vec<Goldilocks>> = tensors
        .iter()
        .map(|t| {
            let d = t.len().trailing_zeros() as usize;
            (0..d).map(|_| rng.field()).collect()
        })
        .collect();
    let claimed: Vec<Goldilocks> = tensors
        .iter()
        .zip(&points)
        .map(|(t, p)| mle::eval(t, p))
        .collect();

    // --- horizontal batching OFF: per-tensor commit + open + verify ---
    println!("== per-tensor commit+open (batching OFF) ==");
    let t0 = Instant::now();
    let mut commit_t = Duration::ZERO;
    let mut open_t = Duration::ZERO;
    let mut verify_t = Duration::ZERO;
    let mut ok = true;
    let mut proof_bytes = 0usize;
    let mut commit_bytes = 0usize;
    for (t, p) in tensors.iter().zip(&points) {
        let d = t.len().trailing_zeros() as usize;
        let pow = Whir::required_pow_bits(d, SECURITY);
        let whir = Whir::with_params(d, SECURITY, pow);
        let t1 = Instant::now();
        let (c, pd, proto) = whir.commit(t);
        commit_t += t1.elapsed();
        commit_bytes += bincode::serialize(&c).unwrap().len();
        let t1 = Instant::now();
        let (proof, ev) = whir.open(pd, &proto, p);
        open_t += t1.elapsed();
        proof_bytes += bincode::serialize(&proof).unwrap().len();
        let t1 = Instant::now();
        ok &= whir
            .verify(&c, &proof, &proto, p)
            .map(|v| v == ev)
            .unwrap_or(false);
        verify_t += t1.elapsed();
    }
    let total_off = t0.elapsed();
    let off_bytes = commit_bytes + proof_bytes;
    println!(
        "  commit {commit_t:?}  open {open_t:?}  verify {verify_t:?}  total {total_off:?}  ok={ok}"
    );
    println!(
        "  size: {} commitments + {} proofs = {}",
        mb(commit_bytes),
        mb(proof_bytes),
        mb(off_bytes)
    );

    // --- horizontal batching ON: per-arity batch commit + one batch open ---
    println!("== per-arity batch commit + batch open (batching ON) ==");
    let mut by_arity: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, t) in tensors.iter().enumerate() {
        let d = t.len().trailing_zeros() as usize;
        by_arity.entry(d).or_default().push(i);
    }
    let t0 = Instant::now();
    let mut commit_t = Duration::ZERO;
    let mut prove_t = Duration::ZERO;
    let mut verify_t = Duration::ZERO;
    let mut ok_b = true;
    let mut proof_bytes = 0usize;
    let mut commit_bytes = 0usize;
    let mut n_groups = 0usize;
    for (&arity, idxs) in &by_arity {
        // Same PoW-budget packing as commit_model_weights: num_vars <= 26.
        let eff_group_size = if arity <= 26 { (1usize << (26 - arity)).max(1) } else { 1 };
        for chunk in idxs.chunks(eff_group_size) {
            n_groups += 1;
            let n = chunk.len();
            let num_vars = (n * (1usize << arity))
                .next_power_of_two()
                .trailing_zeros() as usize;
            let pow = Whir::required_pow_bits(num_vars, SECURITY);
            let base_whir = Whir::with_params(arity, SECURITY, pow);
            let refs: Vec<&[Goldilocks]> =
                chunk.iter().map(|&i| tensors[i].as_slice()).collect();
            let t1 = Instant::now();
            let (commitment, prover_data, protocol, batch_whir) = base_whir.commit_batch(&refs);
            commit_t += t1.elapsed();
            commit_bytes += bincode::serialize(&commitment).unwrap().len();
            let batch = BatchCtx {
                commitment,
                prover_data,
                protocol,
                whir: batch_whir,
                num_tables: n,
            };
            let table_idxs: Vec<usize> = (0..n).collect();
            let sub_tensors: Vec<Vec<Goldilocks>> =
                chunk.iter().map(|&i| tensors[i].clone()).collect();
            let sub_points: Vec<Vec<Goldilocks>> =
                chunk.iter().map(|&i| points[i].clone()).collect();
            let sub_claimed: Vec<Goldilocks> = chunk.iter().map(|&i| claimed[i]).collect();
            let t1 = Instant::now();
            let batch_proof = prove_batch_open_committed(
                &batch,
                &table_idxs,
                &sub_tensors,
                &sub_points,
                &sub_claimed,
            )
            .expect("honest batch open");
            prove_t += t1.elapsed();
            proof_bytes += bincode::serialize(&batch_proof).unwrap().len();
            let t1 = Instant::now();
            ok_b &= verify_batch_open_committed(
                &batch.whir,
                &batch.commitment,
                &batch.protocol,
                n,
                &table_idxs,
                &sub_points,
                &sub_claimed,
                &batch_proof,
            );
            verify_t += t1.elapsed();
            println!(
                "  arity {arity:>2}: {n} tensors, num_vars {num_vars}, pow_bits {pow}"
            );
        }
    }
    let total_on = t0.elapsed();
    let on_bytes = commit_bytes + proof_bytes;
    println!(
        "  {n_groups} groups: commit {commit_t:?}  prove {prove_t:?}  verify {verify_t:?}  total {total_on:?}  ok={ok_b}"
    );
    println!(
        "  size: {} commitments + {} batch proofs = {}",
        mb(commit_bytes),
        mb(proof_bytes),
        mb(on_bytes)
    );

    println!();
    println!("== result ==");
    println!("  OFF: time {total_off:?}  size {}", mb(off_bytes));
    println!("  ON : time {total_on:?}  size {}", mb(on_bytes));
    println!(
        "  time speedup: {:.2}x   size ratio (ON/OFF): {:.2}x",
        total_off.as_secs_f64() / total_on.as_secs_f64(),
        on_bytes as f64 / off_bytes as f64
    );
}
