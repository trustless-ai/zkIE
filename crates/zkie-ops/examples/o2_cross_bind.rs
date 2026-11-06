//! O2 cross-bind route comparison on one boundary tensor with many claims.
//!
//! Runs the incumbent per-claim-opening + same_poly path, route A (merge once
//! then one opening) or route B (one multi-point batch opening, no same_poly)
//! for the same tensor/claims, and reports open count, proof bytes, prove and
//! verify time plus all negative checks.
//!
//! Env:
//!   ZKIE_O2_ROUTE     current | a | b (default current)
//!   ZKIE_O2_LOG2      tensor log2 size (default 20)
//!   ZKIE_O2_CLAIMS    number of claims (default 8)
//!   ZKIE_O2_INPUT     optional i32 file read as the boundary tensor
//!   ZKIE_O2_SEED      synthetic tensor/point seed (default 0x0202)
//!   ZKIE_O2_SELFVERIFY=1  optional prover-side opening self-check for a/b
//!   ZKIE_O2_TRACE=1       print the current path's self-verify time

use std::time::Instant;

use zkie_core::common::field::{ChallengeSource, Goldilocks, PrimeCharacteristicRing, XorShift64};
use zkie_core::common::mle;
use zkie_core::common::transcript::Transcript;
use zkie_core::common::weights_io::WeightMmap;
use zkie_core::pcs::committed::commit;
use zkie_core::pcs::whir::{global_open_count, Whir};
use zkie_ops::compose::{
    committed_cross_bind, committed_cross_bind_per_claim, committed_cross_bind_route_a, committed_cross_bind_route_b,
    verify_committed_cross_bind, verify_committed_cross_bind_per_claim, verify_committed_cross_bind_route_a,
    verify_committed_cross_bind_route_b, BoundaryCommit,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let route = std::env::var("ZKIE_O2_ROUTE").unwrap_or_else(|_| "current".to_string());
    let log2 = env_usize("ZKIE_O2_LOG2", 20);
    let nclaims = env_usize("ZKIE_O2_CLAIMS", 8);
    let seed = env_usize("ZKIE_O2_SEED", 0x0202) as u64;
    let len = 1usize << log2;

    let tensor: Vec<Goldilocks> = match std::env::var("ZKIE_O2_INPUT") {
        Ok(path) => {
            let mmap = WeightMmap::open(&path).expect("open boundary input");
            assert!(mmap.len() >= len, "input shorter than 2^{log2}");
            mmap.read_goldilocks(0, len)
        }
        Err(_) => {
            let mut rng = XorShift64::new(seed);
            (0..len).map(|_| rng.field()).collect()
        }
    };

    // The production verifier reconstructs this parameter set from the
    // boundary arity alone; keep the prover on the same set.
    let whir = Whir::try_with_params(log2, 97, 17).expect("valid O2 boundary Whir params");
    let committed = commit(&whir, &tensor);
    let boundary = BoundaryCommit { commitment: committed.commitment.clone(), arity: log2 };
    let other = {
        let mut rng = XorShift64::new(seed ^ 0xdead_beef);
        let f2: Vec<Goldilocks> = (0..len).map(|_| rng.field()).collect();
        commit(&whir, &f2)
    };
    let wrong_boundary = BoundaryCommit { commitment: other.commitment.clone(), arity: log2 };

    let mut rng = XorShift64::new(seed ^ 0x5151);
    let points: Vec<Vec<Goldilocks>> = (0..nclaims)
        .map(|_| (0..log2).map(|_| rng.field()).collect())
        .collect();
    let claims: Vec<(Vec<Goldilocks>, Goldilocks)> =
        points.iter().map(|p| (p.clone(), mle::eval(&tensor, p))).collect();
    let mut bad = claims.clone();
    bad[0].1 += Goldilocks::ONE;
    let mut short = claims.clone();
    short[0].0.pop();

    let opens0 = global_open_count();
    let (bytes, prove_ms, verify_ms, verify_ok, tamper_rejected, tensor_rejected, shape_rejected) =
        match route.as_str() {
            "current" => {
                let t = Instant::now();
                let mut tp = Transcript::new(b"zkie-committed-binds-v1");
                tp.absorb_u64(seed);
                let proof = committed_cross_bind_per_claim(&whir, &committed, &tensor, &claims, &mut tp)
                    .expect("current prove");
                let prove_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let ok = verify_committed_cross_bind_per_claim(&boundary, &claims, &proof, &mut tv);
                let verify_ms = t.elapsed().as_secs_f64() * 1e3;
                let bytes = bincode::serialize(&proof).expect("serialize current");
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let tamper = !verify_committed_cross_bind_per_claim(&boundary, &bad, &proof, &mut tv);
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let wrong = !verify_committed_cross_bind_per_claim(&wrong_boundary, &claims, &proof, &mut tv);
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let shape = !verify_committed_cross_bind_per_claim(&boundary, &short, &proof, &mut tv);
                (bytes, prove_ms, verify_ms, ok, tamper, wrong, shape)
            }
            "a" => {
                let t = Instant::now();
                let mut tp = Transcript::new(b"zkie-committed-binds-v1");
                tp.absorb_u64(seed);
                let proof =
                    committed_cross_bind_route_a(&whir, &committed, &tensor, &claims, &mut tp)
                        .expect("route a prove");
                let prove_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let ok = verify_committed_cross_bind_route_a(&boundary, &claims, &proof, &mut tv);
                let verify_ms = t.elapsed().as_secs_f64() * 1e3;
                let bytes = bincode::serialize(&proof).expect("serialize route a");
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let tamper = !verify_committed_cross_bind_route_a(&boundary, &bad, &proof, &mut tv);
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let wrong =
                    !verify_committed_cross_bind_route_a(&wrong_boundary, &claims, &proof, &mut tv);
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let shape = !verify_committed_cross_bind_route_a(&boundary, &short, &proof, &mut tv);
                (bytes, prove_ms, verify_ms, ok, tamper, wrong, shape)
            }
            "b" => {
                let t = Instant::now();
                let mut tp = Transcript::new(b"zkie-committed-binds-v1");
                tp.absorb_u64(seed);
                let proof =
                    committed_cross_bind(&whir, &committed, &tensor, &claims, &mut tp)
                        .expect("route b prove");
                let prove_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let ok = verify_committed_cross_bind(&boundary, &claims, &proof, &mut tv);
                let verify_ms = t.elapsed().as_secs_f64() * 1e3;
                let bytes = bincode::serialize(&proof).expect("serialize route b");
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let tamper = !verify_committed_cross_bind(&boundary, &bad, &proof, &mut tv);
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let wrong =
                    !verify_committed_cross_bind(&wrong_boundary, &claims, &proof, &mut tv);
                let mut tv = Transcript::new(b"zkie-committed-binds-v1");
                tv.absorb_u64(seed);
                let shape = !verify_committed_cross_bind(&boundary, &short, &proof, &mut tv);
                (bytes, prove_ms, verify_ms, ok, tamper, wrong, shape)
            }
            other => panic!("unknown ZKIE_O2_ROUTE={other}"),
        };
    let opens = global_open_count() - opens0;
    println!(
        "O2 route={route} log2={log2} claims={nclaims} unique=1 tensor_bytes={} opens={opens} proof_bytes={} prove_ms={prove_ms:.3} verify_ms={verify_ms:.3} verify={verify_ok} tamper_rejected={tamper_rejected} wrong_tensor_rejected={tensor_rejected} wrong_shape_rejected={shape_rejected}",
        len * std::mem::size_of::<Goldilocks>(),
        bytes.len()
    );
    assert!(verify_ok && tamper_rejected && tensor_rejected && shape_rejected);
}
