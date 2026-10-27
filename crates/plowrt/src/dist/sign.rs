//! Signed model indexes.
//!
//! A registry is a bucket anyone can read, so what makes a published index
//! trustworthy is a signature, not where it was fetched from. The index names
//! every manifest by digest, each manifest names its files and objsets by
//! digest, and every blob is verified against those, so one signature over
//! `index.json` covers the whole chain of custody.
//!
//! The scheme is Nix's: ed25519, a key named `<name>:<base64>`, a signature
//! `<name>:<base64>` published beside the index as `index.json.sig`. A client
//! configured with trusted keys (`PLOW_TRUSTED_KEYS`) refuses an index that is
//! unsigned or signed by a key it does not trust; a client with none configured
//! verifies nothing, which is how a `file://` mirror and a test work.
//!
//! The ed25519 code is behind the `dist` feature: a serve-only binary fetches
//! nothing and so verifies nothing.

use crate::{Result, RuntimeError};

/// The signature file published beside a model's `index.json`.
pub const INDEX_SIGNATURE: &str = "index.json.sig";

fn dist(msg: String) -> RuntimeError {
    RuntimeError::Dist(msg)
}

/// A named public key, `<name>:<base64 of 32 bytes>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicKey {
    pub name: String,
    pub bytes: [u8; 32],
}

/// The keys a client accepts an index from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Trust(pub Vec<PublicKey>);

impl Trust {
    pub fn none() -> Self {
        Trust(Vec::new())
    }

    /// `PLOW_TRUSTED_KEYS`: comma-separated `<name>:<base64>` entries.
    pub fn parse(list: &str) -> Result<Self> {
        list.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PublicKey::parse)
            .collect::<Result<Vec<_>>>()
            .map(Trust)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

fn split_named(s: &str, what: &str) -> Result<(String, Vec<u8>)> {
    let (name, b64) = s
        .trim()
        .split_once(':')
        .ok_or_else(|| dist(format!("{what}: expected <name>:<base64>, got {s:?}")))?;
    if name.is_empty() || name.contains(char::is_whitespace) {
        return Err(dist(format!("{what}: {name:?} is not a key name")));
    }
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| dist(format!("{what}: {name}: not base64: {e}")))?;
    Ok((name.to_string(), bytes))
}

fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

impl PublicKey {
    pub fn parse(s: &str) -> Result<Self> {
        let (name, bytes) = split_named(s, "public key")?;
        let bytes: [u8; 32] = bytes
            .try_into()
            .map_err(|_| dist(format!("public key {name}: not 32 bytes")))?;
        Ok(PublicKey { name, bytes })
    }
}

impl std::fmt::Display for PublicKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.name, b64(&self.bytes))
    }
}

/// A signature line as published: `<name>:<base64 of 64 bytes>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub name: String,
    pub bytes: [u8; 64],
}

impl Signature {
    pub fn parse(s: &str) -> Result<Self> {
        let (name, bytes) = split_named(s, "signature")?;
        let bytes: [u8; 64] = bytes
            .try_into()
            .map_err(|_| dist(format!("signature by {name}: not 64 bytes")))?;
        Ok(Signature { name, bytes })
    }
}

impl std::fmt::Display for Signature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.name, b64(&self.bytes))
    }
}

/// Verify `signature` over `message` against the trusted keys. With no
/// trusted key there is nothing to check and `Ok` is returned without a
/// signature; with any, a missing, malformed or untrusted signature is refused.
pub fn verify(message: &[u8], signature: Option<&str>, trust: &Trust) -> Result<()> {
    if trust.is_empty() {
        return Ok(());
    }
    let Some(line) = signature else {
        return Err(dist(format!(
            "the index is unsigned and trusted keys are configured ({}); refusing it",
            trust
                .0
                .iter()
                .map(|k| k.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    };
    let sig = Signature::parse(line)?;
    let Some(key) = trust.0.iter().find(|k| k.name == sig.name) else {
        return Err(dist(format!(
            "the index is signed by {:?}, which is not a trusted key",
            sig.name
        )));
    };
    verify_raw(message, &sig.bytes, &key.bytes).map_err(|e| {
        dist(format!(
            "the index's signature by {} does not verify: {e}",
            key.name
        ))
    })
}

#[cfg(feature = "dist")]
fn verify_raw(message: &[u8], sig: &[u8; 64], key: &[u8; 32]) -> std::result::Result<(), String> {
    let key = ed25519_dalek::VerifyingKey::from_bytes(key).map_err(|e| e.to_string())?;
    let sig = ed25519_dalek::Signature::from_bytes(sig);
    ed25519_dalek::Verifier::verify(&key, message, &sig).map_err(|e| e.to_string())
}

#[cfg(not(feature = "dist"))]
fn verify_raw(_: &[u8], _: &[u8; 64], _: &[u8; 32]) -> std::result::Result<(), String> {
    Err("this plowrt was built without the `dist` feature and cannot verify signatures".into())
}

/// A named signing key: `<name>:<base64 of 64 bytes>` (secret then public),
/// the layout `nix-store --generate-binary-cache-key` writes.
#[cfg(feature = "dist")]
pub struct SecretKey {
    pub name: String,
    key: ed25519_dalek::SigningKey,
}

#[cfg(feature = "dist")]
impl SecretKey {
    /// A fresh key from the OS random source.
    pub fn generate(name: &str) -> Result<Self> {
        if name.is_empty() || name.contains(|c: char| c == ':' || c.is_whitespace()) {
            return Err(dist(format!("{name:?} is not a key name")));
        }
        let seed: [u8; 32] = rand::random();
        Ok(SecretKey {
            name: name.to_string(),
            key: ed25519_dalek::SigningKey::from_bytes(&seed),
        })
    }

    pub fn parse(s: &str) -> Result<Self> {
        let (name, bytes) = split_named(s, "secret key")?;
        let bytes: [u8; 64] = bytes
            .try_into()
            .map_err(|_| dist(format!("secret key {name}: not 64 bytes")))?;
        let key = ed25519_dalek::SigningKey::from_keypair_bytes(&bytes)
            .map_err(|e| dist(format!("secret key {name}: {e}")))?;
        Ok(SecretKey { name, key })
    }

    /// Read the key from a file: one line, `<name>:<base64>`.
    pub fn read(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|source| RuntimeError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(text.trim()).map_err(|e| dist(format!("{}: {e}", path.display())))
    }

    /// The file's contents. Not `Display`: a secret must never reach a log by
    /// accident.
    pub fn to_file_contents(&self) -> String {
        format!("{}:{}\n", self.name, b64(&self.key.to_keypair_bytes()))
    }

    pub fn public(&self) -> PublicKey {
        PublicKey {
            name: self.name.clone(),
            bytes: self.key.verifying_key().to_bytes(),
        }
    }

    pub fn sign(&self, message: &[u8]) -> Signature {
        let sig = ed25519_dalek::Signer::sign(&self.key, message);
        Signature {
            name: self.name.clone(),
            bytes: sig.to_bytes(),
        }
    }
}

#[cfg(all(test, feature = "dist"))]
mod tests {
    use super::*;

    #[test]
    fn a_signature_verifies_only_against_its_own_key_and_its_own_bytes() {
        let k = SecretKey::generate("infervisor-1").unwrap();
        let other = SecretKey::generate("infervisor-2").unwrap();
        let index = b"{\"schema\":\"plow.dist.model.v1\"}";
        let sig = k.sign(index).to_string();
        let trust = Trust(vec![k.public()]);
        verify(index, Some(&sig), &trust).unwrap();
        // Tampered bytes.
        let e = verify(b"{}", Some(&sig), &trust).unwrap_err().to_string();
        assert!(e.contains("does not verify"), "{e}");
        // Unsigned, with trust configured.
        let e = verify(index, None, &trust).unwrap_err().to_string();
        assert!(e.contains("unsigned"), "{e}");
        // Signed by a key the client does not trust.
        let e = verify(index, Some(&other.sign(index).to_string()), &trust)
            .unwrap_err()
            .to_string();
        assert!(e.contains("not a trusted key"), "{e}");
        // No trust configured: nothing to check.
        verify(index, None, &Trust::none()).unwrap();
    }

    #[test]
    fn a_key_round_trips_through_its_file_form() {
        let k = SecretKey::generate("ci").unwrap();
        let again = SecretKey::parse(k.to_file_contents().trim()).unwrap();
        assert_eq!(again.public(), k.public());
        let pk = PublicKey::parse(&k.public().to_string()).unwrap();
        assert_eq!(pk, k.public());
        let trust = Trust::parse(&format!(" {} , {}", pk, pk)).unwrap();
        assert_eq!(trust.0.len(), 2);
        assert!(SecretKey::generate("has space").is_err());
        assert!(PublicKey::parse("nocolon").is_err());
    }
}
