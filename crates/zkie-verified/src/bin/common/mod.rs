use serde::{de::DeserializeOwned, Deserialize};
use std::{fs::File, io::Read, path::Path};

pub const MAX_JSON_BYTES: usize = 4096;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub input: [i32; 2],
    pub context: [u8; 32],
}

pub fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("input file exceeds size limit".into());
    }
    Ok(bytes)
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&read_bounded(
        path,
        MAX_JSON_BYTES,
    )?)?)
}
