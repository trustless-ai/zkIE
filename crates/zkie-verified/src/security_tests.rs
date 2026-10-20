use crate::{air, backend, prove, verify, Error, Model, MAGIC, MAX_PROOF_BYTES};
use p3_air::symbolic::AirLayout;
use p3_field::PrimeCharacteristicRing;
use p3_security::fri::FriRegime;
use p3_uni_stark::{ProvenSecurity, StarkSecurityParams};

fn fixture() -> Model {
    serde_json::from_str(include_str!("../fixtures/model.json")).unwrap()
}

fn encode(proof: &backend::StarkProof) -> Vec<u8> {
    let mut bytes = MAGIC.to_vec();
    bytes.extend(postcard::to_allocvec(proof).unwrap());
    bytes
}

#[test]
fn rejects_corrupt_openings_shapes_and_noncanonical_fields() {
    let m = fixture();
    let (statement, bytes) = prove(&m, [17, -9], [0; 32]).unwrap();
    let decode = || postcard::from_bytes::<backend::StarkProof>(&bytes[MAGIC.len()..]).unwrap();
    let mut bad = decode();
    bad.opened_values.trace_local[0] += backend::F::ONE;
    assert!(verify(&m, m.id().unwrap(), &statement, &encode(&bad)).is_err());
    let mut bad = decode();
    bad.opened_values.trace_local.clear();
    assert!(verify(&m, m.id().unwrap(), &statement, &encode(&bad)).is_err());
    let mut bad = decode();
    bad.opened_values.quotient_chunks.clear();
    assert!(verify(&m, m.id().unwrap(), &statement, &encode(&bad)).is_err());
    for degree in [0, air::TRACE_LOG_ROWS + 1, usize::BITS as usize, usize::MAX] {
        let mut bad = decode();
        bad.degree_bits = degree;
        assert!(verify(&m, m.id().unwrap(), &statement, &encode(&bad)).is_err());
    }
    let mut noncanonical = bytes.clone();
    // First commitment field follows a small postcard vector-length prefix.
    // Setting this whole prefix region to 0xff must fail decoding/verification.
    noncanonical[MAGIC.len()..MAGIC.len() + 16].fill(0xff);
    assert!(verify(&m, m.id().unwrap(), &statement, &noncanonical).is_err());
}

#[test]
fn rejects_malformed_bounded_bytes_without_panicking() {
    let m = fixture();
    let (statement, bytes) = prove(&m, [17, -9], [0; 32]).unwrap();
    let id = m.id().unwrap();
    assert_eq!(
        verify(&m, id, &statement, &vec![0; MAX_PROOF_BYTES + 1]),
        Err(Error::ProofTooLarge)
    );
    for cut in [0, 1, 7, 8, 9, 16, 32, 128, bytes.len() / 2, bytes.len() - 1] {
        assert!(verify(&m, id, &statement, &bytes[..cut]).is_err());
    }
    let mut old_profile = bytes.clone();
    old_profile[7] = b'1';
    assert!(verify(&m, id, &statement, &old_profile).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(verify(&m, id, &statement, &trailing).is_err());
    let mut rng = 0x75ab_931d_3944_ba12u64;
    for len in 0..512 {
        let mut garbage = MAGIC.to_vec();
        for _ in 0..len {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            garbage.push(rng as u8);
        }
        assert!(verify(&m, id, &statement, &garbage).is_err());
    }
}

#[test]
fn concrete_backend_budget_includes_air_and_column_batching() {
    let fri = FriRegime {
        log_blowup: 3,
        num_queries: backend::FRI_QUERIES,
        log_final_poly_len: 0,
        max_log_arity: 1,
        commit_pow_bits: 0,
        query_pow_bits: 0,
    };
    let mut params = StarkSecurityParams::from_air::<backend::F, backend::Challenge, _>(
        fri,
        &air::InferenceAir,
        AirLayout::from_air::<backend::F>(&air::InferenceAir),
        127,
        128,
        1,
    );
    // 70 trace columns plus the two base columns of one quadratic-extension
    // quotient chunk. Do not leave the estimator's optimistic default of one.
    params.num_batched_functions = air::WIDTH + 2;
    assert_eq!(params.num_constraints, 78);
    assert_eq!(params.air_max_constraint_degree, 2);
    let estimate = ProvenSecurity::compute_from_proof(air::TRACE_LOG_ROWS, &params);
    println!(
        "backend bound: unique={} bits, list={} bits, overall={} bits; columns={}",
        estimate.unique_decoding_bits,
        estimate.list_decoding_bits,
        estimate.security_bits(),
        params.num_batched_functions
    );
    assert!(
        estimate.security_bits() >= 96,
        "profile misses the 96-bit backend budget"
    );
}
