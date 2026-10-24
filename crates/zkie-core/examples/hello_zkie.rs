//! `hello_zkie`: the smallest end-to-end zkIE demo — no weights, no dataset.
//!
//! It proves one 4x8 matmul with a single GKR/sum-check reduction, binds the
//! result to a WHIR commitment, and then shows a tampered output being
//! rejected. A second, optional stage proves a table lookup (the LogUp
//! primitive that stands in for a non-linear op). Everything runs in a few
//! milliseconds, so it is the fastest way to see what the stack actually does.
//!
//! ```text
//! cargo run --release -p zkie-core --example hello_zkie
//! ```
//!
//! What the verifier does here, and what it does *not* do:
//!   * it checks the sum-check proof (no recomputation of the product);
//!   * it checks the WHIR opening of the committed `C` at the challenge point,
//!     and that the opened value is the sum-check's claimed value;
//!   * it does *not* bind `A`/`B` to commitments. That binding is what
//!     `zkie_ops::compose::prove_committed_shard_dag` adds for weights; this
//!     example stops before it so the three steps stay visible.
//!
//! At model scale the same three steps are `prove_committed_shard_dag` /
//! `verify_committed_shard_dag` in `zkie-ops`. This is a verifiable-inference
//! (not zero-knowledge) stack: weights and activations are public.

use std::time::{Duration, Instant};

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::{logup_gkr, matmul, mle};
use zkie_core::pcs::whir::Whir;

// WHIR's folding factor is 5, so a commitment needs at least 2^5 slots: keep
// `C` (m x n) at 4 x 8 = 32 field elements.
const M: usize = 4;
const K: usize = 8;
const N: usize = 8;

/// Row-major `A` (`m x k`) transposed into the `k x m` layout the reduction wants.
fn transpose(a: &[Goldilocks], m: usize, k: usize) -> Vec<Goldilocks> {
    let mut at = vec![Goldilocks::ZERO; k * m];
    for i in 0..m {
        for j in 0..k {
            at[j * m + i] = a[i * k + j];
        }
    }
    at
}

/// Plaintext forward pass. The verifier never runs this.
fn forward(a: &[Goldilocks], b: &[Goldilocks], m: usize, k: usize, n: usize) -> Vec<Goldilocks> {
    let mut c = vec![Goldilocks::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = Goldilocks::ZERO;
            for d in 0..k {
                acc += a[i * k + d] * b[d * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// First 8 bytes of the first Merkle root. `Commitment` is a `MerkleCap`, which
/// bincode-encodes as a length-prefixed `Vec<[F; 8]>`, so skip that 8-byte prefix.
fn short_root(commitment: &zkie_core::pcs::whir::Commitment) -> String {
    let bytes = bincode::serialize(commitment).expect("serialize commitment");
    bytes
        .iter()
        .skip(8)
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn main() {
    println!("hello_zkie — C = A @ B (4x8 @ 8x8) + one LogUp lookup, no model weights");
    println!("prove-vs-verify: the checker never runs the forward pass\n");

    // ---- the statement ------------------------------------------------
    let a: Vec<Goldilocks> = (0..M * K)
        .map(|i| Goldilocks::from_u64((i % 5) as u64))
        .collect();
    let b: Vec<Goldilocks> = (0..K * N)
        .map(|i| Goldilocks::from_u64(((i * 3) % 7) as u64))
        .collect();
    let c = forward(&a, &b, M, K, N);
    let at = transpose(&a, M, K);

    // What is being proven, in plain integers (the entries are small sums).
    let rows: Vec<String> = (0..M)
        .map(|i| {
            let row: Vec<String> = (0..N).map(|j| format!("{:?}", c[i * N + j])).collect();
            format!("[{}]", row.join(", "))
        })
        .collect();
    println!("C = A @ B = [{}]\n", rows.join(", "));

    // Prover and verifier derive the same challenges. Production uses the
    // Fiat-Shamir transcript (`common::transcript::Transcript`); a seeded PRNG
    // keeps this example deterministic and readable.
    let mut rng = XorShift64::new(0x5EED);
    let u: Vec<Goldilocks> = (0..M.trailing_zeros() as usize)
        .map(|_| rng.field())
        .collect();
    let v: Vec<Goldilocks> = (0..N.trailing_zeros() as usize)
        .map(|_| rng.field())
        .collect();
    let ch: Vec<Goldilocks> = (0..K.trailing_zeros() as usize)
        .map(|_| rng.field())
        .collect();
    // C is row-major, so column bits are the low bits of the index and row bits
    // the high bits: the evaluation point is (v, u).
    let c_point: Vec<Goldilocks> = v.iter().chain(u.iter()).copied().collect();

    // ---- 1. commit C ---------------------------------------------------
    // `new_testing` is the low-security instance (32-bit security, 10 PoW
    // bits) used for demos/tests; a real proof uses `Whir::new`.
    let whir = Whir::new_testing(M.trailing_zeros() as usize + N.trailing_zeros() as usize);
    let t = Instant::now();
    let (commitment, prover_data, protocol) = whir.commit(&c);
    let commit_ms = ms(t.elapsed());
    println!(
        "[1/6] commit C (32 field elements)        {commit_ms:8.2} ms   root 0x{}…",
        short_root(&commitment)
    );

    // ---- 2. prove -------------------------------------------------------
    let t = Instant::now();
    let proof = matmul::prove(&at, &b, &c, M, K, N, &u, &v, &ch);
    let (opening, opened) = whir.open(prover_data, &protocol, &c_point);
    let prove_ms = ms(t.elapsed());
    println!(
        "[2/6] prove  C = A @ B (GKR over k)       {prove_ms:8.2} ms   claimed C(u,v) = {:?}",
        proof.claimed
    );

    // ---- 3. verify ------------------------------------------------------
    let t = Instant::now();
    let sumcheck_ok = matmul::verify(&proof, &ch);
    let opening_ok = whir
        .verify(&commitment, &opening, &protocol, &c_point)
        .map(|eval| eval == proof.claimed)
        .unwrap_or(false);
    let verify_ms = ms(t.elapsed());
    let accepted = sumcheck_ok && opening_ok;
    // Sanity check for the reader: the opened value is the true MLE of C.
    assert_eq!(opened, proof.claimed);
    assert_eq!(opened, mle::eval(&c, &c_point));
    println!(
        "[3/6] verify sumcheck + opening of C      {verify_ms:8.2} ms   {}",
        if accepted { "PASS" } else { "FAIL" }
    );
    assert!(accepted, "honest proof must verify");

    // ---- 4. tamper: the prover reports a different output ---------------
    let mut c_bad = c.clone();
    c_bad[5] += Goldilocks::ONE;
    let (commitment_bad, pd_bad, protocol_bad) = whir.commit(&c_bad);
    let (opening_bad, _) = whir.open(pd_bad, &protocol_bad, &c_point);
    let tampered_ok = matmul::verify(&proof, &ch)
        && whir
            .verify(&commitment_bad, &opening_bad, &protocol_bad, &c_point)
            .map(|eval| eval == proof.claimed)
            .unwrap_or(false);
    println!(
        "[4/6] tamper one element of C             {:8}      {}",
        "",
        if tampered_ok {
            "PASS (soundness bug!)"
        } else {
            "FAIL as expected"
        }
    );
    assert!(!tampered_ok, "a tampered output must not verify");

    // ---- 5. a non-linear op, via LogUp ----------------------------------
    // The lookup proves "each y[i] equals table[x[i]]" without the verifier
    // recomputing the table read: it checks a fractional sumcheck identity.
    let table: Vec<Goldilocks> = (0..8).map(Goldilocks::from_u64).collect();
    let x: Vec<u32> = vec![3, 1, 0, 2];
    let y: Vec<Goldilocks> = x.iter().map(|&i| table[i as usize]).collect();
    let alpha = rng.field();
    let beta = rng.field();

    let t = Instant::now();
    let lookup = logup_gkr::prove_lookup_fractional(&x, &y, &table, alpha, beta, &mut rng);
    let lookup_prove_ms = ms(t.elapsed());
    let t = Instant::now();
    let lookup_ok = logup_gkr::verify_lookup_fractional(&lookup, &x, &y, &table, alpha, beta);
    let lookup_verify_ms = ms(t.elapsed());
    println!(
        "[5/6] LogUp lookup y[i] = table[x[i]]     {lookup_prove_ms:8.2} ms prove / {lookup_verify_ms:5.2} ms verify   {}",
        if lookup_ok { "PASS" } else { "FAIL" }
    );
    assert!(lookup_ok, "honest lookup must verify");

    let mut y_bad = y.clone();
    y_bad[0] += Goldilocks::ONE;
    let lookup_tampered =
        logup_gkr::verify_lookup_fractional(&lookup, &x, &y_bad, &table, alpha, beta);
    println!(
        "[6/6] tamper one table read               {:8}      {}",
        "",
        if lookup_tampered {
            "PASS (soundness bug!)"
        } else {
            "FAIL as expected"
        }
    );
    assert!(!lookup_tampered, "a tampered lookup must not verify");

    // ---- summary --------------------------------------------------------
    let matmul_bytes =
        bincode::serialize(&proof).unwrap().len() + bincode::serialize(&opening).unwrap().len();
    let lookup_bytes = bincode::serialize(&lookup).unwrap().len();
    println!("\nproof sizes: matmul {matmul_bytes} B, lookup {lookup_bytes} B");
    println!(
        "timings: commit {commit_ms:.2} ms | prove {prove_ms:.2} ms | verify {verify_ms:.2} ms"
    );
    println!(
        "note: at 32 field elements these numbers are dominated by fixed costs; at model \
         scale verify is ~25% of prove (docs/benchmarks.md)"
    );
    println!("scale-up: cargo run --release -p zkie-models-gpt2 --example bench_committed (needs weights)");
}
