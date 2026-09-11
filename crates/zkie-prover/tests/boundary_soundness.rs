use halo2_proofs::halo2curves::bn256::Fr;
use halo2_proofs::halo2curves::ff::PrimeField;
use zkie_core::chips::poseidon_boundary::{
    commit_boundary_native, poseidon_compress_native, BoundaryDescriptor, BoundaryRole,
};
use zkie_core::fixed_point::I18;
use zkie_prover::{BoundaryClaim, LeafStatement};
use zkie_types::{Digest32, ProofFlavorId};

fn fr_from_be(mut bytes: [u8; 32]) -> Fr {
    bytes.reverse();
    Option::from(Fr::from_repr(bytes.into())).expect("canonical BN254 scalar")
}

fn fr_to_be(value: Fr) -> [u8; 32] {
    let mut bytes: [u8; 32] = value.to_repr().into();
    bytes.reverse();
    bytes
}

#[test]
fn circom_two_input_poseidon_matches_independent_known_answer() {
    let expected = [
        13, 84, 225, 147, 143, 138, 140, 28, 125, 235, 94, 3, 85, 242, 99, 25, 32, 123, 132, 254,
        156, 162, 206, 27, 38, 231, 53, 200, 41, 130, 25, 144,
    ];

    assert_eq!(
        fr_to_be(poseidon_compress_native(
            fr_from_be([1_u8; 32]),
            fr_from_be([2_u8; 32]),
        )),
        expected,
    );
}

#[test]
fn boundary_commitment_binds_metadata_length_and_values() {
    let values = [I18::from_raw(-7), I18::from_raw(11)];
    let input = BoundaryDescriptor::flat_i18(
        BoundaryRole::Input,
        "graph-input:x",
        vec!["edge:0".into()],
        Vec::new(),
        2,
        1_000_000_000_000_000_000,
    )
    .unwrap();
    let baseline = commit_boundary_native(&input, &values).unwrap();

    let output = BoundaryDescriptor::flat_i18(
        BoundaryRole::Output,
        "graph-input:x",
        vec!["edge:0".into()],
        Vec::new(),
        2,
        1_000_000_000_000_000_000,
    )
    .unwrap();
    assert_eq!(baseline, commit_boundary_native(&output, &values).unwrap());
    assert_ne!(
        input.public_binding_fields(),
        output.public_binding_fields()
    );

    let broadcast_output = BoundaryDescriptor::flat_i18(
        BoundaryRole::Output,
        "graph-input:x",
        vec!["edge:0".into(), "edge:1".into()],
        Vec::new(),
        2,
        1_000_000_000_000_000_000,
    )
    .unwrap();
    assert_eq!(
        baseline,
        commit_boundary_native(&broadcast_output, &values).unwrap()
    );
    assert_ne!(
        input.public_binding_fields(),
        broadcast_output.public_binding_fields()
    );

    let other_scale = BoundaryDescriptor::flat_i18(
        BoundaryRole::Input,
        "graph-input:x",
        vec!["edge:0".into()],
        Vec::new(),
        2,
        999,
    )
    .unwrap();
    assert_ne!(
        baseline,
        commit_boundary_native(&other_scale, &values).unwrap()
    );
    assert_ne!(
        baseline,
        commit_boundary_native(&input, &[I18::from_raw(-7), I18::from_raw(12)]).unwrap()
    );
    assert!(commit_boundary_native(&input, &values[..1]).is_err());
}

#[test]
fn leaf_statement_instances_bind_all_identity_and_boundary_claims() {
    let descriptor = BoundaryDescriptor::flat_i18(
        BoundaryRole::Input,
        "graph-input:x",
        Vec::new(),
        Vec::new(),
        2,
        1_000_000_000_000_000_000,
    )
    .unwrap();
    let commitment =
        commit_boundary_native(&descriptor, &[I18::from_raw(1), I18::from_raw(2)]).unwrap();
    let statement = LeafStatement::new(
        4,
        "shard-4".into(),
        Digest32::new([1; 32]),
        Digest32::new([2; 32]),
        Digest32::new([3; 32]),
        Digest32::new([4; 32]),
        ProofFlavorId::parse("halo2-kzg-bn256-shplonk-v1").unwrap(),
        Digest32::new([5; 32]),
        vec![BoundaryClaim::new(descriptor, commitment).unwrap()],
        Vec::new(),
        vec![I18::from_raw(17)],
    )
    .unwrap();
    let baseline = statement.instances();

    let changed = statement
        .clone()
        .with_partition_digest(Digest32::new([9; 32]));
    assert_ne!(baseline, changed.instances());
    let encoded = statement.encode().unwrap();
    assert_eq!(LeafStatement::decode(&encoded).unwrap(), statement);
    let mut trailing = encoded;
    trailing.push(0);
    assert!(LeafStatement::decode(&trailing).is_err());
}
