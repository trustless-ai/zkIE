//! Versioned leaf public statement shared by proving and independent verification.

use halo2_proofs::halo2curves::bn256::Fr;
use halo2_proofs::halo2curves::ff::PrimeField;
use zkie_core::chips::poseidon_boundary::{
    BoundaryCommitmentError, BoundaryDescriptor, BoundaryRole,
};
use zkie_core::field_convert::i64_to_fr;
use zkie_core::fixed_point::I18;
use zkie_types::{Digest32, ProofFlavorId};

const MAGIC: &[u8] = b"zkie.leaf-statement.v1\0";
pub const LEAF_STATEMENT_SCHEMA_VERSION: u32 = 2;
const MAX_STATEMENT_BYTES: usize = 1 << 20;
const MAX_CLAIMS: usize = 1 << 12;
const MAX_TEXT_BYTES: usize = 1 << 12;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryClaim {
    descriptor: BoundaryDescriptor,
    commitment: Fr,
}

impl BoundaryClaim {
    pub fn new(descriptor: BoundaryDescriptor, commitment: Fr) -> Result<Self, LeafStatementError> {
        descriptor.validate()?;
        Ok(Self {
            descriptor,
            commitment,
        })
    }
    pub fn descriptor(&self) -> &BoundaryDescriptor {
        &self.descriptor
    }
    pub fn commitment(&self) -> Fr {
        self.commitment
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafStatement {
    shard_id: u64,
    shard_name: String,
    circuit_digest: Digest32,
    partition_digest: Digest32,
    model_digest: Digest32,
    weights_digest: Digest32,
    proof_flavor: ProofFlavorId,
    verification_key_digest: Digest32,
    input_claims: Vec<BoundaryClaim>,
    output_claims: Vec<BoundaryClaim>,
    public_weights: Vec<I18>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeafStatementError {
    EmptyOrOversizedText,
    TooManyClaims,
    WrongBoundaryRole,
    Oversized,
    Malformed,
    UnsupportedVersion,
    InvalidField,
    InvalidDescriptor(BoundaryCommitmentError),
}

impl std::fmt::Display for LeafStatementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for LeafStatementError {}
impl From<BoundaryCommitmentError> for LeafStatementError {
    fn from(value: BoundaryCommitmentError) -> Self {
        Self::InvalidDescriptor(value)
    }
}

impl LeafStatement {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shard_id: u64,
        shard_name: String,
        circuit_digest: Digest32,
        partition_digest: Digest32,
        model_digest: Digest32,
        weights_digest: Digest32,
        proof_flavor: ProofFlavorId,
        verification_key_digest: Digest32,
        input_claims: Vec<BoundaryClaim>,
        output_claims: Vec<BoundaryClaim>,
        public_weights: Vec<I18>,
    ) -> Result<Self, LeafStatementError> {
        if shard_name.is_empty() || shard_name.len() > MAX_TEXT_BYTES {
            return Err(LeafStatementError::EmptyOrOversizedText);
        }
        if input_claims
            .len()
            .checked_add(output_claims.len())
            .ok_or(LeafStatementError::TooManyClaims)?
            > MAX_CLAIMS
        {
            return Err(LeafStatementError::TooManyClaims);
        }
        if input_claims
            .iter()
            .any(|claim| claim.descriptor.role() != BoundaryRole::Input)
            || output_claims
                .iter()
                .any(|claim| claim.descriptor.role() != BoundaryRole::Output)
        {
            return Err(LeafStatementError::WrongBoundaryRole);
        }
        if public_weights.len() > MAX_CLAIMS * 256 {
            return Err(LeafStatementError::Oversized);
        }
        Ok(Self {
            shard_id,
            shard_name,
            circuit_digest,
            partition_digest,
            model_digest,
            weights_digest,
            proof_flavor,
            verification_key_digest,
            input_claims,
            output_claims,
            public_weights,
        })
    }

    pub fn with_partition_digest(mut self, digest: Digest32) -> Self {
        self.partition_digest = digest;
        self
    }
    pub fn with_model_digest(mut self, digest: Digest32) -> Self {
        self.model_digest = digest;
        self
    }
    pub fn with_weights_digest(mut self, digest: Digest32) -> Self {
        self.weights_digest = digest;
        self
    }
    pub fn with_proof_flavor(mut self, flavor: ProofFlavorId) -> Self {
        self.proof_flavor = flavor;
        self
    }
    pub fn with_verification_key_digest(mut self, digest: Digest32) -> Self {
        self.verification_key_digest = digest;
        self
    }
    pub fn with_public_weight(
        mut self,
        index: usize,
        value: I18,
    ) -> Result<Self, LeafStatementError> {
        *self
            .public_weights
            .get_mut(index)
            .ok_or(LeafStatementError::Malformed)? = value;
        Ok(self)
    }
    pub fn with_input_claim(
        mut self,
        index: usize,
        claim: BoundaryClaim,
    ) -> Result<Self, LeafStatementError> {
        if claim.descriptor.role() != BoundaryRole::Input {
            return Err(LeafStatementError::WrongBoundaryRole);
        }
        *self
            .input_claims
            .get_mut(index)
            .ok_or(LeafStatementError::Malformed)? = claim;
        Ok(self)
    }
    pub fn with_output_claim(
        mut self,
        index: usize,
        claim: BoundaryClaim,
    ) -> Result<Self, LeafStatementError> {
        if claim.descriptor.role() != BoundaryRole::Output {
            return Err(LeafStatementError::WrongBoundaryRole);
        }
        *self
            .output_claims
            .get_mut(index)
            .ok_or(LeafStatementError::Malformed)? = claim;
        Ok(self)
    }
    pub fn input_claims(&self) -> &[BoundaryClaim] {
        &self.input_claims
    }
    pub fn output_claims(&self) -> &[BoundaryClaim] {
        &self.output_claims
    }
    pub fn public_weights(&self) -> &[I18] {
        &self.public_weights
    }
    pub fn shard_id(&self) -> u64 {
        self.shard_id
    }
    pub fn shard_name(&self) -> &str {
        &self.shard_name
    }
    pub fn circuit_digest(&self) -> Digest32 {
        self.circuit_digest
    }
    pub fn partition_digest(&self) -> Digest32 {
        self.partition_digest
    }
    pub fn model_digest(&self) -> Digest32 {
        self.model_digest
    }
    pub fn weights_digest(&self) -> Digest32 {
        self.weights_digest
    }
    pub fn proof_flavor(&self) -> &ProofFlavorId {
        &self.proof_flavor
    }
    pub fn verification_key_digest(&self) -> Digest32 {
        self.verification_key_digest
    }
    pub fn instance_prefix(&self) -> Vec<Fr> {
        let mut values = vec![
            Fr::from(LEAF_STATEMENT_SCHEMA_VERSION as u64),
            Fr::from(self.shard_id),
        ];
        values.extend(string_limbs("shard", &self.shard_name));
        for digest in [
            self.circuit_digest,
            self.partition_digest,
            self.model_digest,
            self.weights_digest,
        ] {
            values.extend(digest_limbs(digest));
        }
        values.extend(string_limbs("proof-flavor", self.proof_flavor.as_str()));
        values.extend(digest_limbs(self.verification_key_digest));
        values.push(Fr::from(self.input_claims.len() as u64));
        values
    }

    /// Exact Halo2 public-instance order: protocol, shard identity, circuit,
    /// partition, model, weights, flavor, VK, then each exact descriptor
    /// binding followed by its role-neutral input/output value commitment.
    pub fn instances(&self) -> Vec<Fr> {
        let mut values = self.instance_prefix();
        for claim in &self.input_claims {
            values.extend(claim.descriptor.public_binding_fields());
            values.push(claim.commitment());
        }
        values.push(Fr::from(self.output_claims.len() as u64));
        for claim in &self.output_claims {
            values.extend(claim.descriptor.public_binding_fields());
            values.push(claim.commitment());
        }
        values.push(Fr::from(self.public_weights.len() as u64));
        values.extend(
            self.public_weights
                .iter()
                .map(|value| i64_to_fr(value.raw())),
        );
        values
    }

    pub fn encode(&self) -> Result<Vec<u8>, LeafStatementError> {
        if self.encoded_len()? > MAX_STATEMENT_BYTES {
            return Err(LeafStatementError::Oversized);
        }
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&LEAF_STATEMENT_SCHEMA_VERSION.to_le_bytes());
        out.extend_from_slice(&self.shard_id.to_le_bytes());
        put_string(&mut out, &self.shard_name)?;
        for digest in [
            self.circuit_digest,
            self.partition_digest,
            self.model_digest,
            self.weights_digest,
        ] {
            out.extend_from_slice(digest.as_bytes());
        }
        put_string(&mut out, self.proof_flavor.as_str())?;
        out.extend_from_slice(self.verification_key_digest.as_bytes());
        put_claims(&mut out, &self.input_claims)?;
        put_claims(&mut out, &self.output_claims)?;
        out.extend_from_slice(&(self.public_weights.len() as u32).to_le_bytes());
        for value in &self.public_weights {
            out.extend_from_slice(&value.raw().to_le_bytes());
        }
        if out.len() > MAX_STATEMENT_BYTES {
            return Err(LeafStatementError::Oversized);
        }
        Ok(out)
    }

    fn encoded_len(&self) -> Result<usize, LeafStatementError> {
        let mut len = MAGIC.len()
            + 4
            + 8
            + 4
            + self.shard_name.len()
            + 4 * 32
            + 4
            + self.proof_flavor.as_str().len()
            + 32
            + 4
            + 4
            + 4;
        for claim in self.input_claims.iter().chain(&self.output_claims) {
            let descriptor = claim.descriptor();
            len = len
                .checked_add(1 + 4 + descriptor.register_id().len() + 4 + 4 + 8 + 8 + 32)
                .ok_or(LeafStatementError::Oversized)?;
            for value in descriptor
                .edge_ids()
                .iter()
                .chain(descriptor.graph_output_names())
            {
                len = len
                    .checked_add(4 + value.len())
                    .ok_or(LeafStatementError::Oversized)?;
            }
        }
        len.checked_add(
            self.public_weights
                .len()
                .checked_mul(8)
                .ok_or(LeafStatementError::Oversized)?,
        )
        .ok_or(LeafStatementError::Oversized)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, LeafStatementError> {
        if bytes.len() > MAX_STATEMENT_BYTES || !bytes.starts_with(MAGIC) {
            return Err(LeafStatementError::Malformed);
        }
        let mut cursor = Decoder {
            bytes,
            offset: MAGIC.len(),
        };
        if cursor.u32()? != LEAF_STATEMENT_SCHEMA_VERSION {
            return Err(LeafStatementError::UnsupportedVersion);
        }
        let shard_id = cursor.u64()?;
        let shard_name = cursor.string()?;
        let circuit_digest = cursor.digest()?;
        let partition_digest = cursor.digest()?;
        let model_digest = cursor.digest()?;
        let weights_digest = cursor.digest()?;
        let proof_flavor =
            ProofFlavorId::parse(cursor.string()?).map_err(|_| LeafStatementError::Malformed)?;
        let verification_key_digest = cursor.digest()?;
        let input_claims = cursor.claims(BoundaryRole::Input)?;
        let output_claims = cursor.claims(BoundaryRole::Output)?;
        let weight_count =
            usize::try_from(cursor.u32()?).map_err(|_| LeafStatementError::Malformed)?;
        if weight_count > MAX_CLAIMS * 256 {
            return Err(LeafStatementError::Oversized);
        }
        let public_weights = (0..weight_count)
            .map(|_| {
                Ok(I18::from_raw(i64::from_le_bytes(
                    cursor.take(8)?.try_into().unwrap(),
                )))
            })
            .collect::<Result<Vec<_>, LeafStatementError>>()?;
        if cursor.offset != bytes.len() {
            return Err(LeafStatementError::Malformed);
        }
        Self::new(
            shard_id,
            shard_name,
            circuit_digest,
            partition_digest,
            model_digest,
            weights_digest,
            proof_flavor,
            verification_key_digest,
            input_claims,
            output_claims,
            public_weights,
        )
    }
}

fn digest_limbs(digest: Digest32) -> [Fr; 2] {
    [
        limb(&digest.as_bytes()[..16]),
        limb(&digest.as_bytes()[16..]),
    ]
}
fn string_limbs(domain: &str, value: &str) -> [Fr; 2] {
    let mut input = b"zkie.leaf-string.v1\0".to_vec();
    input.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    input.extend_from_slice(domain.as_bytes());
    input.extend_from_slice(&(value.len() as u64).to_le_bytes());
    input.extend_from_slice(value.as_bytes());
    digest_limbs(Digest32::new(*blake3::hash(&input).as_bytes()))
}
fn limb(bytes: &[u8]) -> Fr {
    let mut repr = [0; 32];
    repr[..bytes.len()].copy_from_slice(bytes);
    Option::from(Fr::from_repr(repr.into())).expect("128-bit limb is canonical")
}
fn put_string(out: &mut Vec<u8>, value: &str) -> Result<(), LeafStatementError> {
    if value.is_empty() || value.len() > MAX_TEXT_BYTES {
        return Err(LeafStatementError::EmptyOrOversizedText);
    }
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}
fn put_claims(out: &mut Vec<u8>, claims: &[BoundaryClaim]) -> Result<(), LeafStatementError> {
    out.extend_from_slice(&(claims.len() as u32).to_le_bytes());
    for claim in claims {
        let d = claim.descriptor();
        out.push(match d.role() {
            BoundaryRole::Input => 1,
            BoundaryRole::Output => 2,
        });
        put_string(out, d.register_id())?;
        put_strings(out, d.edge_ids())?;
        put_strings(out, d.graph_output_names())?;
        out.extend_from_slice(&(d.element_count() as u64).to_le_bytes());
        out.extend_from_slice(&d.quantization_scale().to_le_bytes());
        out.extend_from_slice(claim.commitment().to_repr().as_ref());
    }
    Ok(())
}
fn put_strings(out: &mut Vec<u8>, values: &[String]) -> Result<(), LeafStatementError> {
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    for value in values {
        put_string(out, value)?;
    }
    Ok(())
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Decoder<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], LeafStatementError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(LeafStatementError::Malformed)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(LeafStatementError::Malformed)?;
        self.offset = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32, LeafStatementError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, LeafStatementError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, LeafStatementError> {
        let len = usize::try_from(self.u32()?).map_err(|_| LeafStatementError::Malformed)?;
        if len == 0 || len > MAX_TEXT_BYTES {
            return Err(LeafStatementError::EmptyOrOversizedText);
        }
        std::str::from_utf8(self.take(len)?)
            .map(str::to_owned)
            .map_err(|_| LeafStatementError::Malformed)
    }
    fn digest(&mut self) -> Result<Digest32, LeafStatementError> {
        Ok(Digest32::new(self.take(32)?.try_into().unwrap()))
    }
    fn strings(&mut self) -> Result<Vec<String>, LeafStatementError> {
        let count = usize::try_from(self.u32()?).map_err(|_| LeafStatementError::Malformed)?;
        if count > MAX_CLAIMS {
            return Err(LeafStatementError::TooManyClaims);
        }
        (0..count).map(|_| self.string()).collect()
    }
    fn claims(
        &mut self,
        expected_role: BoundaryRole,
    ) -> Result<Vec<BoundaryClaim>, LeafStatementError> {
        let count = usize::try_from(self.u32()?).map_err(|_| LeafStatementError::Malformed)?;
        if count > MAX_CLAIMS {
            return Err(LeafStatementError::TooManyClaims);
        }
        (0..count)
            .map(|_| {
                let role = match self.take(1)?[0] {
                    1 => BoundaryRole::Input,
                    2 => BoundaryRole::Output,
                    _ => return Err(LeafStatementError::Malformed),
                };
                if role != expected_role {
                    return Err(LeafStatementError::WrongBoundaryRole);
                }
                let register = self.string()?;
                let edges = self.strings()?;
                let outputs = self.strings()?;
                let count =
                    usize::try_from(self.u64()?).map_err(|_| LeafStatementError::Malformed)?;
                let scale = self.u64()?;
                let descriptor =
                    BoundaryDescriptor::flat_i18(role, register, edges, outputs, count, scale)?;
                let repr: [u8; 32] = self.take(32)?.try_into().unwrap();
                let commitment = Option::from(Fr::from_repr(repr.into()))
                    .ok_or(LeafStatementError::InvalidField)?;
                BoundaryClaim::new(descriptor, commitment)
            })
            .collect()
    }
}
