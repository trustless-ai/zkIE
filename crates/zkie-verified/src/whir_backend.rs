//! Shared, explicitly parameterized PCS for the witness-free reduction route.
use crate::backend::F;
use p3_field::extension::BinomialExtensionField;
pub(crate) type Challenge = BinomialExtensionField<F, 5>;
type E = Challenge;
use p3_challenger::{CanObserve, DuplexChallenger};
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_goldilocks::{default_goldilocks_poseidon2_16, Poseidon2Goldilocks};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::SuffixProver;
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::pcs::prover::WhirProver;

type Perm = Poseidon2Goldilocks<16>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type Mmcs = MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, Hash, Compress, 2, 8>;
pub(crate) type Challenger = DuplexChallenger<F, Perm, 16, 8>;
pub(crate) type Pcs =
    WhirProver<E, F, Radix2DFTSmallBatch<F>, Mmcs, Challenger, SuffixProver<F, E>>;
pub(crate) type Commitment = <Pcs as MultilinearPcs<E, Challenger>>::Commitment;
pub(crate) type Opening = <Pcs as MultilinearPcs<E, Challenger>>::Proof;
#[cfg(feature = "prover")]
pub(crate) type Data = <Pcs as MultilinearPcs<E, Challenger>>::ProverData;

pub(crate) fn pcs(vars: usize) -> Result<Pcs, crate::Error> {
    let folding = FoldingFactor::Constant(5);
    let (rounds, _) = folding
        .compute_number_of_rounds(vars)
        .map_err(|_| crate::Error::InvalidModel)?;
    let params = ProtocolParameters {
        security_level: 128,
        pow_bits: 0,
        folding_factor: folding,
        soundness_type: SecurityAssumption::UniqueDecoding,
        starting_log_inv_rate: 1,
        round_log_inv_rates: (0..rounds).map(|i| 1 + 4 * (i + 1)).collect(),
    };
    let config = WhirConfig::<E, F, Challenger>::new(vars, params)
        .map_err(|_| crate::Error::InvalidModel)?;
    let perm = default_goldilocks_poseidon2_16();
    let mmcs = Mmcs::new(Hash::new(perm.clone()), Compress::new(perm), 0);
    let dft = Radix2DFTSmallBatch::new(1 << config.max_fft_size());
    Ok(Pcs::new(config, dft, mmcs))
}

pub(crate) fn challenger(pcs: &Pcs, digest: &[u8]) -> Challenger {
    let mut c = Challenger::new(default_goldilocks_poseidon2_16());
    let mut domain = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut domain);
    domain.observe_domain_separator(&mut c);
    use p3_field::PrimeCharacteristicRing;
    c.observe_slice(&digest.iter().map(|&b| F::from_u8(b)).collect::<Vec<_>>());
    c
}

pub(crate) fn protocol(vars: usize) -> OpeningProtocol {
    OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(vars, 1),
        std::iter::once(OpeningBatch::new(vec![0], vec![])).collect(),
    )])
}

/// Our reductions use little-endian bit coordinates; the upstream PCS uses BE.
pub(crate) fn point(little: &[E], vars: usize) -> Point<E> {
    use p3_field::PrimeCharacteristicRing;
    let mut coordinates = little.to_vec();
    coordinates.resize(vars, E::ZERO);
    coordinates.reverse();
    Point::new(coordinates)
}
