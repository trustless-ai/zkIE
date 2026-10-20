use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{extension::BinomialExtensionField, Field};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::{default_goldilocks_poseidon2_16, Goldilocks, Poseidon2Goldilocks};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;

pub(crate) const FRI_QUERIES: usize = 128;

pub(crate) type F = Goldilocks;
type Perm = Poseidon2Goldilocks<16>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, Hash, Compress, 2, 8>;
pub(crate) type Challenge = BinomialExtensionField<F, 2>;
type ChallengeMmcs = ExtensionMmcs<F, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;
type Pcs = TwoAdicFriPcs<F, Radix2DitParallel<F>, ValMmcs, ChallengeMmcs>;
pub(crate) type Config = StarkConfig<Pcs, Challenge, Challenger>;
pub(crate) type StarkProof = p3_uni_stark::Proof<Config>;

/// One pinned profile; no caller-controlled testing/security downgrade.
pub(crate) fn config() -> Config {
    let perm = default_goldilocks_poseidon2_16();
    let mmcs = ValMmcs::new(Hash::new(perm.clone()), Compress::new(perm.clone()), 0);
    let fri = FriParameters {
        log_blowup: 3,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: FRI_QUERIES,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
        mmcs: ChallengeMmcs::new(mmcs.clone()),
    };
    let pcs = Pcs::new(Radix2DitParallel::default(), mmcs, fri);
    Config::new(pcs, Challenger::new(perm))
}
