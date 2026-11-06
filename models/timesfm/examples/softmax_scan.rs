//! Scan every Softmax op in the real TimesFM graph and report the maximum
//! absolute row sum S. Used to size the committed Softmax remainder window.
//!
//! Usage: cargo run --release -p zkie-models-timesfm --example softmax_scan

use zkie_models_timesfm::build_timesfm;
use zkie_ops::compose::{evaluate_ops, Op, Store};

fn main() {
    let mut store = Store::new();
    let mut ops = Vec::new();
    let _ = build_timesfm(&mut store, &mut ops, "models/timesfm/weights", 16);
    evaluate_ops(&mut store, &ops);
    let mut max_s: i64 = 0;
    let mut max_at = (0usize, 0usize);
    let mut min_s: i64 = 0;
    let mut count = 0usize;
    for (op_index, op) in ops.iter().enumerate() {
        let Op::Softmax { e, m, n, .. } = op else { continue };
        let ev = store.get(*e);
        if ev.len() != m * n {
            eprintln!("softmax op {op_index}: e len {} != {}x{}", ev.len(), m, n);
            std::process::exit(2);
        }
        count += 1;
        for i in 0..*m {
            let s: i64 = (0..*n)
                .map(|j| zkie_core::common::fixed_point::to_i64(ev[i * n + j]))
                .sum();
            if s.abs() > max_s {
                max_s = s.abs();
                max_at = (op_index, i);
            }
            if s < min_s { min_s = s; }
        }
    }
    println!("timesfm softmax_ops={count} max_abs_S={max_s} at_op_row={max_at:?} min_S={min_s}");
    let bit = if max_s <= 1 { 0 } else {
        (usize::BITS - ((max_s - 1) as u64).leading_zeros()) as usize
    };
    println!("timesfm ceil_log2(max_abs_S)={bit} next_pow2={}", 1u64 << bit);

    // StableSoftmaxIndex range scan: mdiff = rmax - masked,
    // tr_w = trunc - w, diff = trunc - idx (the three 22-bit ranges).
    let mut ssi_ops = 0usize;
    let mut max_mdiff = 0i64;
    let mut max_tr_w = 0i64;
    let mut max_diff = 0i64;
    let mut max_mdiff_at = (0usize, 0usize);
    let mut max_tr_w_at = (0usize, 0usize);
    for (op_index, op) in ops.iter().enumerate() {
        let Op::StableSoftmaxIndex { x, mask, out, offset, table_len, m, n, .. } = op else { continue };
        ssi_ops += 1;
        let xv = store.get(*x);
        let mv = store.get(*mask);
        let idxs = &store.idx[*out];
        for i in 0..*m {
            let mut rmax = i64::MIN;
            for j in 0..*n {
                let at = i * n + j;
                let masked = zkie_core::common::fixed_point::to_i64(xv[at])
                    .checked_add(zkie_core::common::fixed_point::to_i64(mv[at])).unwrap();
                if masked > rmax { rmax = masked; }
            }
            for j in 0..*n {
                let at = i * n + j;
                let masked = zkie_core::common::fixed_point::to_i64(xv[at])
                    .checked_add(zkie_core::common::fixed_point::to_i64(mv[at])).unwrap();
                let w = masked.checked_sub(rmax).unwrap().checked_add(*offset as i64).unwrap();
                let trunc = w.max(0);
                let idx = idxs[at] as i64;
                let mdiff = rmax.checked_sub(masked).unwrap();
                let tr_w = trunc.checked_sub(w).unwrap();
                let diff = trunc.checked_sub(idx).unwrap();
                if mdiff > max_mdiff { max_mdiff = mdiff; max_mdiff_at = (op_index, at); }
                if tr_w > max_tr_w { max_tr_w = tr_w; max_tr_w_at = (op_index, at); }
                if diff > max_diff { max_diff = diff; }
                let _ = table_len;
            }
        }
    }
    println!("timesfm ssi_ops={ssi_ops} max_mdiff={max_mdiff} at={max_mdiff_at:?} max_tr_w={max_tr_w} at={max_tr_w_at:?} max_diff={max_diff}");
    let bits = |v: i64| if v <= 1 { 0 } else {
        (usize::BITS - ((v - 1) as u64).leading_zeros()) as usize };
    println!("timesfm ssi_bits mdiff={} tr_w={} diff={}",
        bits(max_mdiff), bits(max_tr_w), bits(max_diff));

    // LNC bounded ranges: |mean| (signed 16-bit raw + 16-bit rem) and the
    // signed 32-bit pin ob = out - b + 2^31.
    let mut lnc_ops = 0usize;
    let mut max_abs_mean = 0i64;
    let mut max_ob = 0i64;
    for (op_index, op) in ops.iter().enumerate() {
        let Op::LayerNormCentered { x, b, out, m, d, n_real, .. } = op else { continue };
        lnc_ops += 1;
        let xv = store.get(*x);
        let bv = store.get(*b);
        let ov = store.get(*out);
        for i in 0..*m {
            let s: i64 = (0..*n_real).map(|j| {
                zkie_core::common::fixed_point::to_i64(xv[i * d + j])
            }).sum();
            let mean = (s + (*n_real as i64) / 2).div_euclid(*n_real as i64);
            if mean.abs() > max_abs_mean { max_abs_mean = mean.abs(); }
            for j in 0..*d {
                let at = i * d + j;
                let ob = zkie_core::common::fixed_point::to_i64(ov[at])
                    .checked_sub(zkie_core::common::fixed_point::to_i64(bv[at])).unwrap()
                    .checked_add(1i64 << 31).unwrap();
                if ob > max_ob { max_ob = ob; }
            }
            let _ = op_index;
        }
    }
    println!("timesfm lnc_ops={lnc_ops} max_abs_mean={max_abs_mean} max_ob={max_ob} ob_limit={}",
        1u64 << 32);

    // Argmax witness ranges: gap = d2 (top minus second logit) and
    // tie = gap - before + sel.
    let mut arg_ops = 0usize;
    let mut max_gap = 0i64;
    let mut max_tie = 0i64;
    let mut max_gap_at = (0usize, 0usize);
    for (op_index, op) in ops.iter().enumerate() {
        let Op::Argmax { sel, out, d2, m, n, .. } = op else { continue };
        arg_ops += 1;
        let selv = store.get(*sel);
        let d2v = store.get(*d2);
        let outs = &store.idx[*out];
        for i in 0..*m {
            let oi = outs[i] as usize;
            for j in 0..*n {
                let at = i * n + j;
                let gap = zkie_core::common::fixed_point::to_i64(d2v[at]);
                let tie = gap - if j < oi { 1 } else { 0 }
                    + zkie_core::common::fixed_point::to_i64(selv[at]);
                if gap > max_gap { max_gap = gap; max_gap_at = (op_index, at); }
                if tie > max_tie { max_tie = tie; }
            }
        }
    }
    println!("timesfm argmax_ops={arg_ops} max_gap={max_gap} at={max_gap_at:?} max_tie={max_tie}");
    println!("timesfm argmax_bits gap={} tie={}", bits(max_gap), bits(max_tie));
}
