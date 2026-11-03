//! Exportable collective allocations. KV VMM properties deliberately stay private to KV.

use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};

use super::*;

const POSIX_FD: i32 = 1;

pub(crate) struct IpcRegion {
    be: Arc<CudaBackend>,
    pub base: u64,
    pub bytes: u64,
    pub alignment: u64,
    handle: u64,
    mapped: bool,
    leak: bool,
}

impl IpcRegion {
    pub fn create(be: Arc<CudaBackend>, bytes: u64) -> Result<(Self, OwnedFd)> {
        be.bind()?;
        let prop = CUmemAllocationProp {
            type_: MEM_ALLOCATION_TYPE_PINNED,
            requested_handle_types: POSIX_FD,
            location: CUmemLocation {
                type_: MEM_LOCATION_TYPE_DEVICE,
                id: be.device_ordinal as i32,
            },
            win32_handle_meta_data: std::ptr::null_mut(),
            alloc_flags: [0; 8],
        };
        let mut granularity = 0usize;
        be.check(
            unsafe { (be.api.cuMemGetAllocationGranularity)(&mut granularity, &prop, 0) },
            "cuMemGetAllocationGranularity(IPC)",
        )?;
        let alignment = (granularity as u64).max(2 << 20);
        if granularity == 0 || !alignment.is_power_of_two() {
            return Err(RuntimeError::Device(
                "invalid IPC allocation granularity".into(),
            ));
        }
        let bytes = bytes
            .checked_add(alignment - 1)
            .map(|n| n / alignment * alignment)
            .filter(|&n| n > 0 && n <= usize::MAX as u64)
            .ok_or_else(|| RuntimeError::Device("IPC allocation size overflow".into()))?;
        let mut region = Self {
            be,
            base: 0,
            bytes,
            alignment,
            handle: 0,
            mapped: false,
            leak: false,
        };
        region.be.check(
            unsafe { (region.be.api.cuMemCreate)(&mut region.handle, bytes as usize, &prop, 0) },
            "cuMemCreate(IPC)",
        )?;
        region.map()?;
        let mut fd = -1i32;
        region.be.check(
            unsafe {
                (region.be.api.cuMemExportToShareableHandle)(
                    (&mut fd as *mut i32).cast(),
                    region.handle,
                    POSIX_FD,
                    0,
                )
            },
            "cuMemExportToShareableHandle",
        )?;
        // The driver transfers a new descriptor; set CLOEXEC before it can leave this scope.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
            return Err(RuntimeError::Device(format!(
                "IPC FD_CLOEXEC: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok((region, fd))
    }

    pub fn import(
        be: Arc<CudaBackend>,
        fd: BorrowedFd<'_>,
        bytes: u64,
        alignment: u64,
    ) -> Result<Self> {
        if alignment < 2 << 20
            || !alignment.is_power_of_two()
            || bytes == 0
            || bytes % alignment != 0
            || bytes > usize::MAX as u64
        {
            return Err(RuntimeError::Device("invalid IPC mapping geometry".into()));
        }
        be.bind()?;
        let mut region = Self {
            be,
            base: 0,
            bytes,
            alignment,
            handle: 0,
            mapped: false,
            leak: false,
        };
        region.be.check(
            unsafe {
                (region.be.api.cuMemImportFromShareableHandle)(
                    &mut region.handle,
                    fd.as_raw_fd() as usize as *mut c_void,
                    POSIX_FD,
                )
            },
            "cuMemImportFromShareableHandle",
        )?;
        region.map()?;
        Ok(region)
    }

    fn map(&mut self) -> Result<()> {
        self.be.check(
            unsafe {
                (self.be.api.cuMemAddressReserve)(
                    &mut self.base,
                    self.bytes as usize,
                    self.alignment as usize,
                    0,
                    0,
                )
            },
            "cuMemAddressReserve(IPC)",
        )?;
        self.be.check(
            unsafe { (self.be.api.cuMemMap)(self.base, self.bytes as usize, 0, self.handle, 0) },
            "cuMemMap(IPC)",
        )?;
        self.mapped = true;
        let access = CUmemAccessDesc {
            location: CUmemLocation {
                type_: MEM_LOCATION_TYPE_DEVICE,
                id: self.be.device_ordinal as i32,
            },
            flags: MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        self.be.check(
            unsafe { (self.be.api.cuMemSetAccess)(self.base, self.bytes as usize, &access, 1) },
            "cuMemSetAccess(IPC)",
        )
    }

    pub fn leak_on_drop(&mut self) {
        self.leak = true;
    }
}

impl Drop for IpcRegion {
    fn drop(&mut self) {
        // A failed drain does not prove that remote addresses are no longer in use locally.
        if self.leak || self.be.ipc_must_retain() || self.be.synchronize().is_err() {
            tracing::warn!(va = self.base, "retaining IPC mapping after failed drain");
            return;
        }
        unsafe {
            if self.mapped {
                let rc = (self.be.api.cuMemUnmap)(self.base, self.bytes as usize);
                if rc != 0 {
                    tracing::warn!(rc, "cuMemUnmap(IPC) failed; retaining handle and VA");
                    return;
                }
            }
            if self.handle != 0 {
                let rc = (self.be.api.cuMemRelease)(self.handle);
                if rc != 0 {
                    tracing::warn!(rc, "cuMemRelease(IPC) failed");
                }
            }
            if self.base != 0 {
                let rc = (self.be.api.cuMemAddressFree)(self.base, self.bytes as usize);
                if rc != 0 {
                    tracing::warn!(rc, "cuMemAddressFree(IPC) failed");
                }
            }
        }
    }
}

impl CudaBackend {
    pub(crate) fn ipc_must_retain(&self) -> bool {
        self.is_poisoned() || self.ipc_failed_drain.load(Ordering::Relaxed)
    }

    pub(crate) fn stream_write_value32(
        &self,
        stream: &CudaStream,
        dst: u64,
        value: u32,
    ) -> Result<()> {
        self.bind()?;
        self.check(
            unsafe { (self.api.cuStreamWriteValue32)(stream.raw as CUstream, dst, value, 0) },
            "cuStreamWriteValue32(IPC)",
        )
    }
}
