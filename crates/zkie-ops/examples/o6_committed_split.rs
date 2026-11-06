//! O6 measurement: committed-mode witness construction vs the old
//! scratch-proof-then-reprove path.
//!
//! The direct calls build one Scatter and one Gather op in the committed
//! mode. Before the split they generated fractional proofs on scratch
//! transcripts (discarded by the committed section); after the split they
//! build only the witness columns. The end-to-end line proves a small
//! committed shard DAG containing both ops.
//!
//! Run on both trees with the same command and compare the op_* and e2e
//! numbers. Proof bytes are expected to change (the committed protocol
//! version changes); the witness columns and the final proof subtasks do not.

use zkie_core::common::field::{Goldilocks, PrimeCharacteristicRing};
use zkie_core::common::fixed_point::from_i64;
use zkie_core::common::logup_gkr::{
    merge_multiplicity, prove_lookup_table, prove_lookup_witness_batch_columns,
};
use zkie_core::common::transcript::Transcript;
use zkie_core::pcs::whir::Whir;
use zkie_ops::compose::{prove_committed_shard_dag, FracMode, Op, Store};
use zkie_ops::gather::{gather_forward, prove_scatter_add_with_mode};
use zkie_ops::scatter::{prove_scatter_with_mode, scatter_forward};

fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
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

fn main() {
    // Scatter: m tokens x n experts, k routed experts per token.
    let (m, n, k) = (1024usize, 1024usize, 4usize);
    let s = m * k;
    let h = 64usize;
    let mut sel = vec![Goldilocks::ZERO; m * n];
    for t in 0..m {
        for j in 0..k {
            let e = (t * 7 + j * 13) % n;
            sel[t * n + e] = Goldilocks::ONE;
        }
    }
    let (token_id, expert_of) = scatter_forward(&sel, m, n);
    assert_eq!(token_id.len(), s);
    let weighted: Vec<Goldilocks> = (0..s * h).map(|i| from_i64((i as i64 % 9) - 4)).collect();
    let gate: Vec<Goldilocks> = (0..s).map(|i| from_i64((i as i64 % 5) - 2)).collect();
    let out = gather_forward(&weighted, &token_id, &gate, m, h);

    // Direct committed-mode witness construction.
    let mut tsp = Transcript::new(b"o6-split-scatter");
    tsp.absorb_u64(0x06);
    let t0 = std::time::Instant::now();
    let sp = prove_scatter_with_mode(&sel, &token_id, &expert_of, m, n, &mut tsp, FracMode::Committed);
    let scatter_ms = t0.elapsed().as_secs_f64() * 1e3;
    let scatter_cols = (sp.perm.a.len(), sp.perm.b.len());

    let mut tgp = Transcript::new(b"o6-split-gather");
    tgp.absorb_u64(0x06);
    let t1 = std::time::Instant::now();
    let gp = prove_scatter_add_with_mode(
        &weighted,
        &token_id,
        &gate,
        &out,
        m,
        h,
        &mut tgp,
        FracMode::Committed,
    );
    let gather_ms = t1.elapsed().as_secs_f64() * 1e3;
    let gather_cols = (gp.weighted.len(), gp.gate.len(), gp.out.len(), gp.token_id.len());

    // End-to-end committed shard DAG (one op per shard).
    let mut store = Store::new();
    let sel_t = store.push(sel);
    let tok_t = store.push_idx(token_id.clone());
    let exp_t = store.push_idx(expert_of.clone());
    let weighted_t = store.push(weighted.clone());
    let gate_t = store.push(gate.clone());
    let out_t = store.push(out.clone());
    let tokens_t = store.push_idx(Vec::new());
    let argmax_t = store.push_idx(Vec::new());
    let ops = vec![
        Op::Scatter {
            sel: sel_t,
            token_id: tok_t,
            expert_of: exp_t,
            m,
            n,
        },
        Op::Gather {
            weighted: weighted_t,
            token_id: tok_t,
            gate: gate_t,
            out: out_t,
            m,
            h,
        },
    ];
    let whir = Whir::new_testing(5);
    let rss_before = status_kb("VmRSS:");
    let t2 = std::time::Instant::now();
    let proof = prove_committed_shard_dag(&mut store, &ops, 1, &whir, &[], tokens_t, argmax_t);
    let e2e_ms = t2.elapsed().as_secs_f64() * 1e3;
    let rss_after = status_kb("VmRSS:");
    let hwm = status_kb("VmHWM:");
    let bytes = bincode::serialize(&proof).expect("serialize committed proof");

    // Lookup/table-side fractional work: the old committed shard stage
    // generated both proofs and then re-proved them after the commitments.
    // The split keeps only the column build here. Time the old direct calls
    // so the discarded work is recorded next to the scatter/gather numbers.
    let lut_len = 4096usize;
    let idx_len = 65536usize;
    let table_vals: Vec<Goldilocks> =
        (0..lut_len).map(|i| Goldilocks::from_u64((i as u64 % 97) + 100)).collect();
    let idx_vals: Vec<u32> = (0..idx_len).map(|i| (i % lut_len) as u32).collect();
    let y_vals: Vec<Goldilocks> = idx_vals.iter().map(|&i| table_vals[i as usize]).collect();
    let xs: Vec<&[u32]> = vec![idx_vals.as_slice()];
    let ys: Vec<&[Goldilocks]> = vec![y_vals.as_slice()];
    let alpha = Goldilocks::from_u64(1_234_567);
    let beta = Goldilocks::from_u64(7_654_321);
    let mut tlw = Transcript::new(b"o6-lookup-witness");
    let tl1 = std::time::Instant::now();
    let _wp = prove_lookup_witness_batch_columns(&xs, &ys, alpha, beta, &mut tlw);
    let lookup_witness_us = tl1.elapsed().as_secs_f64() * 1e6;
    let mult = merge_multiplicity(&xs, table_vals.len());
    let mut tlt = Transcript::new(b"o6-lookup-table");
    let tl2 = std::time::Instant::now();
    let _tp = prove_lookup_table(&mult, &table_vals, alpha, beta, &mut tlt);
    let lookup_table_us = tl2.elapsed().as_secs_f64() * 1e6;

    println!(
        "O6-SPLIT m={} n={} k={} h={} scatter_ms={:.3} scatter_a={} scatter_b={} gather_ms={:.3} gather_cols={:?} e2e_ms={:.1} proof_len={} proof_hash={:016x} rss_before_kb={} rss_after_kb={} hwm_kb={} lookup_witness_us={:.1} lookup_table_us={:.1}",
        m, n, k, h, scatter_ms, scatter_cols.0, scatter_cols.1, gather_ms, gather_cols,
        e2e_ms, bytes.len(), fnv(&bytes), rss_before, rss_after, hwm,
        lookup_witness_us, lookup_table_us
    );
}
