use std::{error::Error, fmt, str::FromStr};

use serde::{Deserialize, Serialize};

/// A fixed-width content digest, serialized as 32 bytes and displayed as lowercase hex.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Digest32([u8; 32]);

impl Digest32 {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for Digest32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for Digest32 {
    type Err = DigestParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() != 64 {
            return Err(DigestParseError::InvalidLength);
        }
        let mut bytes = [0; 32];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            bytes[index] = hex_byte(pair).ok_or(DigestParseError::InvalidHex)?;
        }
        Ok(Self(bytes))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DigestParseError {
    InvalidLength,
    InvalidHex,
}

impl fmt::Display for DigestParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidLength => "digest must contain exactly 64 hexadecimal characters",
            Self::InvalidHex => "digest must use lowercase hexadecimal characters",
        })
    }
}

impl Error for DigestParseError {}

fn hex_byte(pair: &[u8]) -> Option<u8> {
    Some((hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

macro_rules! ascii_identifier {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn parse(value: impl Into<String>) -> Result<Self, IdentifierError> {
                let value = value.into();
                if value.is_empty() {
                    return Err(IdentifierError::Empty);
                }
                if !value.is_ascii() {
                    return Err(IdentifierError::NonAscii);
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = IdentifierError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::parse(value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                String::deserialize(deserializer)
                    .and_then(|value| Self::parse(value).map_err(serde::de::Error::custom))
            }
        }
    };
}

ascii_identifier!(ProofFlavorId);
ascii_identifier!(ExecutionBackendId);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentifierError {
    Empty,
    NonAscii,
}

impl fmt::Display for IdentifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "identifier must not be empty",
            Self::NonAscii => "identifier must contain only ASCII characters",
        })
    }
}

impl Error for IdentifierError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelVisibility {
    PublicModel,
    PrivateModel,
}

/// All inputs which bind a run and determine whether its artifacts can be reused.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunIdentity {
    pub model_graph_digest: Digest32,
    pub weights_digest: Digest32,
    pub compiler_digest: Digest32,
    pub isa_digest: Digest32,
    pub quantization_digest: Digest32,
    pub partition_plan_digest: Digest32,
    pub aggregation_plan_digest: Digest32,
    pub proof_flavor: ProofFlavorId,
    pub model_visibility: ModelVisibility,
    pub aggregation_fan_in: u32,
    pub public_input_schema_version: u32,
}

impl RunIdentity {
    /// Computes an unambiguous BLAKE3 digest over every identity field.
    pub fn canonical_digest(&self) -> Digest32 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"zkie.run-identity.v1");
        for digest in [
            self.model_graph_digest,
            self.weights_digest,
            self.compiler_digest,
            self.isa_digest,
            self.quantization_digest,
            self.partition_plan_digest,
            self.aggregation_plan_digest,
        ] {
            hasher.update(digest.as_bytes());
        }
        update_string(&mut hasher, self.proof_flavor.as_str());
        hasher.update(&[match self.model_visibility {
            ModelVisibility::PublicModel => 0,
            ModelVisibility::PrivateModel => 1,
        }]);
        hasher.update(&self.aggregation_fan_in.to_le_bytes());
        hasher.update(&self.public_input_schema_version.to_le_bytes());
        Digest32::new(*hasher.finalize().as_bytes())
    }
}

fn update_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}
