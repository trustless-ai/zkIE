use crate::{backend::F, Error};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const VERSION: u32 = 2;
const DOMAIN: &[u8] = b"zkie-verified/dense2x2-q4-relu/stark-v2\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelId(pub [u8; 32]);

impl ModelId {
    pub fn to_hex(self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn from_hex(s: &str) -> Result<Self, Error> {
        if s.len() != 64 || !s.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(Error::ModelIdentity);
        }
        let mut id = [0; 32];
        for (i, byte) in id.iter_mut().enumerate() {
            *byte =
                u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| Error::ModelIdentity)?;
        }
        Ok(Self(id))
    }
}

/// Public verification material. Fixed dimensions are [layer][output][input].
/// Q4 weights in [-8,8], Q4 biases in [-16,16]. No secret setup or model files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub version: u32,
    pub fractional_bits: u8,
    pub weights: [[[i32; 2]; 2]; 2],
    pub biases: [[i32; 2]; 2],
}

impl Model {
    pub fn validate(&self) -> Result<(), Error> {
        if self.version != VERSION
            || self.fractional_bits != 4
            || self
                .weights
                .iter()
                .flatten()
                .flatten()
                .any(|w| !(-8..=8).contains(w))
            || self
                .biases
                .iter()
                .flatten()
                .any(|b| !(-16..=16).contains(b))
        {
            return Err(Error::InvalidModel);
        }
        Ok(())
    }

    /// Canonical, versioned binary identity, independent of JSON formatting.
    pub fn id(&self) -> Result<ModelId, Error> {
        self.validate()?;
        let mut hash = Sha256::new();
        hash.update(DOMAIN);
        hash.update(self.version.to_le_bytes());
        hash.update([self.fractional_bits]);
        for n in self
            .weights
            .iter()
            .flatten()
            .flatten()
            .chain(self.biases.iter().flatten())
        {
            hash.update(n.to_le_bytes());
        }
        Ok(ModelId(hash.finalize().into()))
    }

    pub(crate) fn public_values(&self, statement: &Statement) -> Vec<F> {
        // The backend absorbs all public values before deriving its challenges.
        // Bind metadata not otherwise used by arithmetic constraints as a digest.
        let mut hash = Sha256::new();
        hash.update(DOMAIN);
        hash.update(statement.version.to_le_bytes());
        hash.update(statement.model_id.0);
        hash.update(statement.context);
        for n in statement.input.iter().chain(statement.output.iter()) {
            hash.update(n.to_le_bytes());
        }
        let mut values: Vec<F> = hash.finalize().iter().map(|&b| F::from_u8(b)).collect();
        values.extend(
            self.weights
                .iter()
                .flatten()
                .flatten()
                .chain(self.biases.iter().flatten())
                .chain(statement.input.iter())
                .chain(statement.output.iter())
                .copied()
                .map(signed),
        );
        debug_assert_eq!(values.len(), crate::air::PUBLIC_VALUES);
        values
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statement {
    pub version: u32,
    pub model_id: ModelId,
    /// Caller-selected request/session binding, not a timestamp or authenticity proof.
    pub context: [u8; 32],
    pub input: [i32; 2],
    pub output: [i32; 2],
}

impl Statement {
    pub fn validate(&self) -> Result<(), Error> {
        if self.version != VERSION
            || self.input.iter().any(|x| !(-128..=127).contains(x))
            || self.output.iter().any(|y| !(-256..=255).contains(y))
        {
            return Err(Error::InvalidStatement);
        }
        Ok(())
    }
}

pub(crate) fn signed(n: i32) -> F {
    if n < 0 {
        F::ZERO - F::from_u64((-i64::from(n)) as u64)
    } else {
        F::from_u32(n as u32)
    }
}
