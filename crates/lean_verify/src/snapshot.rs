use std::io::Read;
#[cfg(target_os = "linux")]
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::VerifyError;

pub(crate) struct Snapshot {
    pub(crate) digest: [u8; 32],
    pub(crate) path: PathBuf,
    #[cfg(target_os = "linux")]
    _image: std::fs::File,
}

impl Snapshot {
    pub(crate) fn capture(path: &Path) -> Result<Self, VerifyError> {
        let resolved = if path.components().count() == 1 {
            std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
                .map(|dir| dir.join(path))
                .find(|file| file.is_file())
                .ok_or_else(|| VerifyError::BinaryNotFound(path.display().to_string()))?
        } else {
            path.to_path_buf()
        };
        let mut source = std::fs::File::open(&resolved)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if source.metadata()?.permissions().mode() & 0o111 == 0 {
                return Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied).into());
            }
        }
        let mut bytes = Vec::new();
        source.read_to_end(&mut bytes)?;
        let digest = Sha256::digest(&bytes).into();
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::{AsRawFd, FromRawFd};
            // Keep the fd inherited: script interpreters must reopen /proc/self/fd/N.
            let fd =
                unsafe { libc::memfd_create(c"plow-verify".as_ptr(), libc::MFD_ALLOW_SEALING) };
            if fd < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            let mut image = unsafe { std::fs::File::from_raw_fd(fd) };
            image.write_all(&bytes)?;
            let seals =
                libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
            if unsafe { libc::fcntl(image.as_raw_fd(), libc::F_ADD_SEALS, seals) } < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok(Self {
                digest,
                path: PathBuf::from(format!("/proc/self/fd/{fd}")),
                _image: image,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(Self {
                digest,
                path: resolved,
            })
        }
    }

    pub(crate) fn sha256(&self) -> String {
        self.digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn snapshot_cannot_change_after_source_replacement() {
        let directory =
            std::env::temp_dir().join(format!("plow-verifier-snapshot-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("verify");
        let original = b"#!/bin/sh\nprintf original";
        std::fs::write(&path, original).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        let image = Snapshot::capture(&path).unwrap();
        std::fs::write(&path, b"#!/bin/sh\nprintf replaced").unwrap();
        assert!(std::fs::write(&image.path, b"mutation").is_err());
        let output = std::process::Command::new(&image.path).output().unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout, b"original");
        assert_eq!(image.digest, <[u8; 32]>::from(Sha256::digest(original)));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
