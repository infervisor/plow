//! Nix archive (NAR) serialisation of a directory tree, and the store path
//! name its hash yields. The identity of a distributed bundle
//! (`docs/arch/27-content-addressed-distribution.md`).
//!
//! A NAR covers exactly what the loader can observe: each regular file's
//! bytes and executable bit, each directory's entries in byte order of their
//! names, and each symlink's target *string*. Nothing else — no mtime, owner,
//! mode bits beyond `x`, hard-link structure or extended attributes — so the
//! same tree written twice, anywhere, hashes the same. A symlink contributes
//! its target and not what it points at, which is what lets a bundle name a
//! checkpoint (`checkpoint -> ../../snapshots/<org>--<repo>@<commit>`) without
//! the hash depending on the weights being present.
//!
//! Self-contained rather than a crate: the format is a page of the Nix
//! thesis (figure 5.2), the two crates that implement it either are not
//! published (snix's `nix-compat` is git-only) or restrict paths to UTF-8 and
//! size a fixed buffer by the longest file name (`nix-nar`), and the writer
//! here streams, so a 4 GB sidecar is hashed without being read into memory.
//!
//! The store path is Nix's own formula with the store directory fixed to
//! `/nix/store`, so `nix store add --dry-run <dir>` on a build host prints the
//! same name `plowrt` computes: an operator can cross-check a published path
//! with stock tooling and no plow binary.

use std::io::{self, Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

const MAGIC: &[u8] = b"nix-archive-1";

/// Write `root`'s NAR serialisation to `out`.
pub fn pack(root: &Path, out: &mut dyn Write) -> io::Result<()> {
    string(out, MAGIC)?;
    node(root, out)
}

/// The sha256 of `root`'s NAR, computed as it is serialised.
pub fn nar_sha256(root: &Path) -> io::Result<[u8; 32]> {
    struct HashWriter(Sha256);
    impl Write for HashWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.update(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut w = HashWriter(Sha256::new());
    pack(root, &mut w)?;
    Ok(w.0.finalize().into())
}

/// `<32 base32 chars>-<name>`: the store path a tree with this NAR hash is
/// kept under, by the formula `nix-store --add` uses for a recursively
/// hashed source with the store directory `/nix/store`.
///
/// The 20-byte fingerprint is a fold of a second sha256, not the NAR hash
/// itself, which is why the NAR hash travels separately in the narinfo.
pub fn store_path(nar_sha256: &[u8; 32], name: &str) -> Result<String, String> {
    if name.is_empty()
        || name.len() > 211
        || name.starts_with('.')
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"+-._?=".contains(&b))
    {
        return Err(format!(
            "{name:?} is not a store path name: 1..211 of [A-Za-z0-9+-._?=], not starting with `.`"
        ));
    }
    let fingerprint = format!("source:sha256:{}:/nix/store:{name}", hex(nar_sha256));
    let h = Sha256::digest(fingerprint.as_bytes());
    let mut folded = [0u8; 20];
    for (i, b) in h.iter().enumerate() {
        folded[i % 20] ^= b;
    }
    Ok(format!("{}-{name}", nix_base32(&folded)))
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Nix's base32: its own alphabet (no `e o u t`), and the bits taken from the
/// end of the input first, so it matches `nix hash path --base32` and the
/// hash part of a store path rather than RFC 4648.
pub fn nix_base32(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 32] = b"0123456789abcdfghijklmnpqrsvwxyz";
    let len = (bytes.len() * 8).div_ceil(5);
    let mut out = String::with_capacity(len);
    for n in (0..len).rev() {
        let b = n * 5;
        let (i, j) = (b / 8, b % 8);
        // u16: a shift by 8 - j with j == 0 is a shift of the whole byte out,
        // which is what the C original does under int promotion.
        let lo = (bytes[i] as u16) >> j;
        let hi = if i + 1 < bytes.len() {
            (bytes[i + 1] as u16) << (8 - j)
        } else {
            0
        };
        out.push(ALPHABET[((lo | hi) & 0x1f) as usize] as char);
    }
    out
}

fn string(out: &mut dyn Write, s: &[u8]) -> io::Result<()> {
    out.write_all(&(s.len() as u64).to_le_bytes())?;
    out.write_all(s)?;
    pad(out, s.len() as u64)
}

fn pad(out: &mut dyn Write, len: u64) -> io::Result<()> {
    let rem = (len % 8) as usize;
    if rem != 0 {
        out.write_all(&[0u8; 8][..8 - rem])?;
    }
    Ok(())
}

fn node(path: &Path, out: &mut dyn Write) -> io::Result<()> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let m = std::fs::symlink_metadata(path)?;
    string(out, b"(")?;
    string(out, b"type")?;
    let ft = m.file_type();
    if ft.is_symlink() {
        string(out, b"symlink")?;
        string(out, b"target")?;
        string(out, std::fs::read_link(path)?.as_os_str().as_bytes())?;
    } else if ft.is_dir() {
        string(out, b"directory")?;
        let mut entries: Vec<(Vec<u8>, std::path::PathBuf)> = std::fs::read_dir(path)?
            .map(|e| e.map(|e| (e.file_name().as_bytes().to_vec(), e.path())))
            .collect::<io::Result<_>>()?;
        // Byte order of the name, which is what Nix sorts by; a locale-aware
        // or case-folded order would hash the same tree differently.
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, child) in entries {
            string(out, b"entry")?;
            string(out, b"(")?;
            string(out, b"name")?;
            string(out, &name)?;
            string(out, b"node")?;
            node(&child, out)?;
            string(out, b")")?;
        }
    } else if ft.is_file() {
        string(out, b"regular")?;
        if m.permissions().mode() & 0o111 != 0 {
            string(out, b"executable")?;
            string(out, b"")?;
        }
        string(out, b"contents")?;
        // The length is written before the bytes, so a file that grows or
        // shrinks while it is read would corrupt the archive silently; the
        // copy is checked against the length promised.
        let len = m.size();
        out.write_all(&len.to_le_bytes())?;
        let mut f = std::fs::File::open(path)?;
        let copied = io::copy(&mut (&mut f).take(len), out)?;
        if copied != len || f.read(&mut [0u8])? != 0 {
            return Err(io::Error::other(format!(
                "{}: changed while being archived ({len} bytes promised)",
                path.display()
            )));
        }
        pad(out, len)?;
    } else {
        return Err(io::Error::other(format!(
            "{}: not a regular file, directory or symlink; a NAR cannot carry it",
            path.display()
        )));
    }
    string(out, b")")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn tmp(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "plow-nar-{}-{}-{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    /// The tree `nix hash path` / `nix store add --dry-run` were run on:
    /// an executable, a dangling relative symlink of the checkpoint form,
    /// an empty directory, a nested object, and names that sort differently
    /// by byte and by case (`Zeta` before `bin`).
    fn fixture(root: &Path) {
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::create_dir_all(root.join("hsaco")).unwrap();
        std::fs::create_dir_all(root.join("empty")).unwrap();
        std::fs::write(root.join("bundle.json"), "{}\n").unwrap();
        std::fs::write(root.join("bin/run"), "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(root.join("bin/run"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        std::fs::write(root.join("hsaco/a.cubin"), "AAAA").unwrap();
        std::fs::write(root.join("z-last"), "z").unwrap();
        std::fs::write(root.join("Zeta"), "Z").unwrap();
        std::os::unix::fs::symlink(
            "../snapshots/google--gemma-4-12b-it@842da3794eaa0b77d5f08bae87a17459d91ff475",
            root.join("checkpoint"),
        )
        .unwrap();
    }

    // Expected values from nix 2.35.2 on the tree `fixture` builds (the same
    // names, bytes, mode and symlink target, created with mkdir/printf/chmod
    // 755/ln -s in a scratch directory named `fixture`):
    //   nix hash path --type sha256 --base16 fixture
    //     a6e98f2f3e8e4609065fa0d606d4a9dd9efe667869c605c9c5e514cf4ad67c2a
    //   nix hash path --type sha256 --base32 fixture
    //     0akwsr5cy575qp4hbik9g1kgx7nxm7a0dmm0bw30jilf7qpqzsd6
    //   nix nar pack fixture | wc -c
    //     1880
    //   nix store add --dry-run --name fixture fixture   (= nix-store --add)
    //     /nix/store/agwa6hp53s58zg7r3ghiwa2kzii1rs40-fixture
    #[test]
    fn the_fixture_hashes_as_nix_hashes_it() {
        let root = tmp("fixture");
        fixture(&root);
        let mut nar = Vec::new();
        pack(&root, &mut nar).unwrap();
        assert_eq!(nar.len(), 1880);
        let h = nar_sha256(&root).unwrap();
        assert_eq!(h, <[u8; 32]>::from(Sha256::digest(&nar)));
        assert_eq!(
            hex(&h),
            "a6e98f2f3e8e4609065fa0d606d4a9dd9efe667869c605c9c5e514cf4ad67c2a"
        );
        assert_eq!(
            nix_base32(&h),
            "0akwsr5cy575qp4hbik9g1kgx7nxm7a0dmm0bw30jilf7qpqzsd6"
        );
        assert_eq!(
            store_path(&h, "fixture").unwrap(),
            "agwa6hp53s58zg7r3ghiwa2kzii1rs40-fixture"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A bundle must hash the same with and without its checkpoint present,
    /// and regardless of when it was written: the symlink target string is
    /// the reference, the snapshot's contents are not part of the bundle.
    #[test]
    fn a_symlink_contributes_its_target_not_what_it_points_at() {
        let base = tmp("symlink");
        let root = base.join("store").join("x-fixture");
        fixture(&root);
        let dangling = nar_sha256(&root).unwrap();

        let snap = base
            .join("store")
            .join("snapshots")
            .join("google--gemma-4-12b-it@842da3794eaa0b77d5f08bae87a17459d91ff475");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("model.safetensors"), vec![7u8; 4096]).unwrap();
        assert!(
            root.join("checkpoint/model.safetensors").is_file(),
            "resolves now"
        );
        assert_eq!(nar_sha256(&root).unwrap(), dangling);

        // Ownership-free metadata the loader cannot see is not identity either.
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(86_400);
        std::fs::File::open(root.join("bundle.json"))
            .unwrap()
            .set_modified(later)
            .unwrap();
        std::fs::set_permissions(root.join("z-last"), std::fs::Permissions::from_mode(0o600))
            .unwrap();
        assert_eq!(nar_sha256(&root).unwrap(), dangling);

        // What the loader can see is.
        std::fs::set_permissions(root.join("z-last"), std::fs::Permissions::from_mode(0o700))
            .unwrap();
        assert_ne!(nar_sha256(&root).unwrap(), dangling, "the executable bit");
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn a_store_path_name_is_validated() {
        let h = [0u8; 32];
        assert!(store_path(&h, "infervisor-kimi-k3-gfx942-mi325x-tp8-32k").is_ok());
        for bad in ["", ".hidden", "a/b", "a b", "ü"] {
            assert!(store_path(&h, bad).is_err(), "{bad:?}");
        }
        let long = "a".repeat(212);
        assert!(store_path(&h, &long).is_err());
    }
}
