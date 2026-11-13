//! A rank group whose ranks are processes: one device per process, ranks exchanging
//! collective partials through one shared, page-locked host region.
//!
//! This is the transport for devices that cannot map each other's memory — MIG instances of
//! one GPU (no peer access, no IPC between instances, and a process sees one instance) — and
//! it runs unchanged on separate GPUs. The device side keeps the in-program collective
//! contract of [`super::tp`]: each rank has a [`PeerLayout`] region (partial slots, `xctr`
//! gates, `xstatus`), the program reads `peer_scratch[r]`, and the gates count
//! system-scope arrivals. Only where the regions live differs: here they are consecutive
//! slices of one `memfd` that every rank maps and registers with its own driver, so each
//! process builds its own `peer_scratch` table from its own mapping.
//!
//! # Reset discipline
//!
//! A single-process group zeroes every rank's counters before any rank launches. Ranks in
//! different processes cannot, so every launch unit (one decode step, one prefill chunk)
//! starts with [`ProcGroup::arm`]: the rank zeroes its OWN `xctr`/`xstatus` lines (its device
//! is idle — the previous unit was synchronized, and no peer signals a gate after the owner
//! passed it) and then waits at a host barrier. No rank launches, so no rank signals a peer,
//! before every rank has zeroed. The barrier fails instead of hanging when a peer process
//! exits or the group is aborted.
//!
//! The control page also carries a hand-off buffer for activations crossing a pipeline stage
//! boundary.

use std::io::{Read, Write};
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use super::tp::{PeerLayout, XCTR_STRIDE};
use crate::{Result, RuntimeError};

const MAGIC: u64 = 0x504c_4f57_5052_4750; // "PLOWPRGP"
/// Most ranks one group carries (control page slots).
pub const MAX_RANKS: usize = 64;
const PAGE: u64 = 4096;

#[repr(C)]
struct Ctrl {
    magic: u64,
    n: u32,
    /// Ranks per barrier subgroup (the tensor-parallel width; ranks `[k*tp, (k+1)*tp)`).
    tp: u32,
    region_stride_pages: u32,
    handoff_pages: u32,
    /// Nonzero once any rank gave up; every later barrier fails at once.
    abort: AtomicU32,
    _pad: u32,
    pid: [AtomicU32; MAX_RANKS],
    arrive: [AtomicU32; MAX_RANKS],
    generation: [AtomicU32; MAX_RANKS],
    /// Launch-plan fingerprint per rank, double-buffered by barrier generation parity.
    plan: [[AtomicU64; MAX_RANKS]; 2],
}

/// The shared mapping: control page, then `n` rank regions, then the stage hand-off buffer.
pub struct ProcGroup {
    fd: OwnedFd,
    base: *mut u8,
    len: usize,
    n: u32,
    tp: u32,
    rank: u32,
    stride: u64,
    handoff_off: u64,
    handoff_bytes: u64,
    layout: Option<PeerLayout>,
}

// SAFETY: the mapping is process-shared memory addressed only through raw pointers whose
// cross-thread use is ordered by the control page's atomics or by device synchronization.
unsafe impl Send for ProcGroup {}
unsafe impl Sync for ProcGroup {}

fn os_err(what: &str) -> RuntimeError {
    RuntimeError::Device(format!("{what}: {}", std::io::Error::last_os_error()))
}

fn round_page(b: u64) -> u64 {
    b.div_ceil(PAGE) * PAGE
}

impl ProcGroup {
    /// Create the group mapping for `n` ranks (the caller is rank 0), barrier subgroups of `tp`
    /// ranks. `layout` sizes each rank region (None = no collectives, e.g. a pure pipeline);
    /// `handoff_bytes` sizes the stage hand-off buffer.
    pub fn create(n: u32, tp: u32, layout: Option<PeerLayout>, handoff_bytes: u64) -> Result<Self> {
        if n < 2 || n as usize > MAX_RANKS || tp == 0 || n % tp != 0 {
            return Err(RuntimeError::Rejected(format!(
                "a process rank group needs 2..={MAX_RANKS} ranks in whole subgroups of {tp}, got {n}"
            )));
        }
        let region = layout.map_or(0, |l| round_page(l.bytes()));
        let handoff = round_page(handoff_bytes);
        let len = PAGE + region * n as u64 + handoff;
        // SAFETY: plain syscalls on a name we own; the result is checked.
        let raw = unsafe { libc::memfd_create(c"plow-rank-group".as_ptr(), 0) };
        if raw < 0 {
            return Err(os_err("memfd_create"));
        }
        // SAFETY: `raw` is a fresh descriptor this function owns.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        // SAFETY: valid fd, non-negative length.
        if unsafe { libc::ftruncate(fd.as_raw_fd(), len as libc::off_t) } != 0 {
            return Err(os_err("ftruncate rank group"));
        }
        let mut g = Self::map(fd, len as usize, 0)?;
        g.n = n;
        g.tp = tp;
        g.stride = region;
        g.handoff_off = PAGE + region * n as u64;
        g.handoff_bytes = handoff;
        g.layout = layout;
        let c = g.ctrl_mut_init();
        c.n = n;
        c.tp = tp;
        c.region_stride_pages = (region / PAGE) as u32;
        c.handoff_pages = (handoff / PAGE) as u32;
        c.magic = MAGIC;
        g.ctrl().pid[0].store(std::process::id(), Ordering::Release);
        Ok(g)
    }

    /// Attach rank `rank` to a group created by [`Self::create`] (the descriptor was
    /// inherited). `layout` must equal the creator's.
    pub fn attach(fd: OwnedFd, rank: u32, layout: Option<PeerLayout>) -> Result<Self> {
        // SAFETY: fstat on an owned fd into a zeroed struct.
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0 {
            return Err(os_err("fstat rank group"));
        }
        let mut g = Self::map(fd, st.st_size as usize, rank)?;
        let (magic, n, tp, stride_pages, handoff_pages) = {
            let c = g.ctrl();
            (c.magic, c.n, c.tp, c.region_stride_pages, c.handoff_pages)
        };
        if magic != MAGIC || rank >= n {
            return Err(RuntimeError::Device(format!(
                "rank group descriptor is not a group of more than {rank} ranks"
            )));
        }
        g.n = n;
        g.tp = tp;
        g.stride = stride_pages as u64 * PAGE;
        g.handoff_off = PAGE + g.stride * n as u64;
        g.handoff_bytes = handoff_pages as u64 * PAGE;
        if layout.map_or(0, |l| round_page(l.bytes())) != g.stride {
            return Err(RuntimeError::Device(
                "rank group region size differs from this rank's packet layout".into(),
            ));
        }
        g.layout = layout;
        g.ctrl().pid[rank as usize].store(std::process::id(), Ordering::Release);
        Ok(g)
    }

    fn map(fd: OwnedFd, len: usize, rank: u32) -> Result<Self> {
        // SAFETY: shared mapping of a descriptor we own; checked against MAP_FAILED.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(os_err("mmap rank group"));
        }
        Ok(Self {
            fd,
            base: p.cast(),
            len,
            n: 0,
            tp: 0,
            rank,
            stride: 0,
            handoff_off: 0,
            handoff_bytes: 0,
            layout: None,
        })
    }

    fn ctrl(&self) -> &Ctrl {
        // SAFETY: the first page is the control block for the mapping's lifetime.
        unsafe { &*(self.base as *const Ctrl) }
    }

    fn ctrl_mut_init(&mut self) -> &mut Ctrl {
        // SAFETY: only the creator calls this, before any other process holds the fd.
        unsafe { &mut *(self.base as *mut Ctrl) }
    }

    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }
    pub fn n(&self) -> u32 {
        self.n
    }
    pub fn rank(&self) -> u32 {
        self.rank
    }
    pub fn tp(&self) -> u32 {
        self.tp
    }
    pub fn layout(&self) -> Option<PeerLayout> {
        self.layout
    }
    /// The whole mapping (what a rank registers with its driver).
    pub fn mapping(&self) -> (*mut u8, usize) {
        (self.base, self.len)
    }
    /// Byte offset of rank `r`'s region within the mapping.
    pub fn region_offset(&self, r: u32) -> u64 {
        PAGE + self.stride * r as u64
    }
    pub fn handoff(&self) -> (*mut u8, u64) {
        // SAFETY: in-bounds offset of the mapping.
        (
            unsafe { self.base.add(self.handoff_off as usize) },
            self.handoff_bytes,
        )
    }

    /// Zero this rank's counter and status lines. Caller guarantees its device is idle.
    pub fn zero_own_xstate(&self) {
        let Some(l) = self.layout else { return };
        let off = self.region_offset(self.rank) + l.xctr_off();
        // SAFETY: in-bounds: the region holds `xstate_bytes` past `xctr_off`.
        unsafe {
            std::ptr::write_bytes(self.base.add(off as usize), 0, l.xstate_bytes() as usize);
        }
        std::sync::atomic::fence(Ordering::SeqCst);
    }

    /// The word at line `id` of this rank's counter region (`id == n_xctr` is the status line).
    pub fn own_xline(&self, id: u32) -> u32 {
        let Some(l) = self.layout else { return 0 };
        let off = self.region_offset(self.rank) + l.xctr_off() + id as u64 * XCTR_STRIDE as u64;
        // SAFETY: in-bounds aligned word, read volatile: devices write it.
        unsafe { std::ptr::read_volatile(self.base.add(off as usize) as *const u32) }
    }

    /// Start of one launch unit: zero own counters, then wait for every rank of this subgroup
    /// and check that all of them are about to launch the same plan (`plan` fingerprint). A
    /// mismatch fails every member before any of them launches.
    pub fn arm(&self, plan: u64, timeout: Duration) -> Result<()> {
        self.zero_own_xstate();
        let parity = self.barrier_with(Some(plan), timeout)?;
        let c = self.ctrl();
        let first = self.rank / self.tp * self.tp;
        for r in first..first + self.tp {
            let p = c.plan[parity][r as usize].load(Ordering::Acquire);
            if p != plan {
                self.abort();
                return Err(RuntimeError::CollectiveBail(format!(
                    "rank {r} is launching a different plan than rank {} ({p:#x} vs {plan:#x})",
                    self.rank
                )));
            }
        }
        Ok(())
    }

    /// Mark the group dead so every rank's next barrier fails immediately.
    pub fn abort(&self) {
        self.ctrl().abort.store(1, Ordering::Release);
    }

    pub fn aborted(&self) -> bool {
        self.ctrl().abort.load(Ordering::Acquire) != 0
    }

    fn peer_gone(&self) -> Option<u32> {
        let c = self.ctrl();
        (0..self.n).find(|&r| {
            let pid = c.pid[r as usize].load(Ordering::Acquire);
            // SAFETY: signal 0 only checks existence.
            r != self.rank && pid != 0 && unsafe { libc::kill(pid as i32, 0) } != 0
        })
    }

    /// Sense-reversing barrier over this rank's subgroup.
    pub fn barrier(&self, timeout: Duration) -> Result<()> {
        self.barrier_with(None, timeout).map(|_| ())
    }

    /// The barrier, publishing `plan` first; returns the generation parity it was published in.
    fn barrier_with(&self, plan: Option<u64>, timeout: Duration) -> Result<usize> {
        let c = self.ctrl();
        if self.aborted() {
            return Err(RuntimeError::CollectiveBail("rank group aborted".into()));
        }
        let sub = (self.rank / self.tp) as usize;
        let gen = c.generation[sub].load(Ordering::Acquire);
        let parity = (gen & 1) as usize;
        if let Some(p) = plan {
            c.plan[parity][self.rank as usize].store(p, Ordering::Release);
        }
        if c.arrive[sub].fetch_add(1, Ordering::AcqRel) + 1 == self.tp {
            c.arrive[sub].store(0, Ordering::Relaxed);
            c.generation[sub].store(gen.wrapping_add(1), Ordering::Release);
            return Ok(parity);
        }
        let t0 = Instant::now();
        let mut spins = 0u32;
        while c.generation[sub].load(Ordering::Acquire) == gen {
            spins = spins.wrapping_add(1);
            if spins % 1024 == 0 {
                if self.aborted() {
                    return Err(RuntimeError::CollectiveBail("rank group aborted".into()));
                }
                if let Some(r) = self.peer_gone() {
                    self.abort();
                    return Err(RuntimeError::CollectiveBail(format!(
                        "rank {r} of the group exited"
                    )));
                }
                if t0.elapsed() > timeout {
                    self.abort();
                    return Err(RuntimeError::CollectiveBail(format!(
                        "rank group barrier timed out after {timeout:?} on rank {}",
                        self.rank
                    )));
                }
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        Ok(parity)
    }
}

impl Drop for ProcGroup {
    fn drop(&mut self) {
        // SAFETY: unmapping the mapping this value created.
        unsafe { libc::munmap(self.base.cast(), self.len) };
    }
}

/// Length-prefixed JSON messages over a stream socket (leader <-> one worker).
pub struct Link {
    s: UnixStream,
}

impl Link {
    pub fn new(s: UnixStream) -> Self {
        Self { s }
    }
    /// Close both directions now: the peer's next read sees end-of-stream.
    pub fn close(&self) {
        let _ = self.s.shutdown(std::net::Shutdown::Both);
    }
    /// The ancillary marker is a separate frame, always consumed by recvmsg, never read().
    pub fn send_fd<T: serde::Serialize>(
        &mut self,
        v: &T,
        fd: Option<BorrowedFd<'_>>,
    ) -> Result<()> {
        let mut marker = [u8::from(fd.is_some())];
        let mut iov = libc::iovec {
            iov_base: marker.as_mut_ptr().cast(),
            iov_len: 1,
        };
        // usize storage keeps cmsghdr aligned.
        let mut control = [0usize; 8];
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        if let Some(fd) = fd {
            msg.msg_control = control.as_mut_ptr().cast();
            // `msg_controllen` and `cmsg_len` are size_t on glibc, socklen_t on musl.
            msg.msg_controllen =
                unsafe { libc::CMSG_SPACE(std::mem::size_of::<RawFd>() as u32) } as _;
            unsafe {
                let h = libc::CMSG_FIRSTHDR(&msg);
                (*h).cmsg_level = libc::SOL_SOCKET;
                (*h).cmsg_type = libc::SCM_RIGHTS;
                (*h).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<RawFd>() as u32) as _;
                std::ptr::write_unaligned(libc::CMSG_DATA(h).cast::<RawFd>(), fd.as_raw_fd());
            }
        }
        loop {
            let n = unsafe { libc::sendmsg(self.s.as_raw_fd(), &msg, libc::MSG_NOSIGNAL) };
            if n == 1 {
                break;
            }
            if n < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(os_err("rank link sendmsg"));
        }
        self.send(v)
    }

    pub fn recv_fd<T: serde::de::DeserializeOwned>(&mut self) -> Result<(T, Option<OwnedFd>)> {
        let mut marker = [0u8];
        let mut iov = libc::iovec {
            iov_base: marker.as_mut_ptr().cast(),
            iov_len: 1,
        };
        let mut control = [0usize; 8];
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        let n = loop {
            msg.msg_controllen = std::mem::size_of_val(&control) as _;
            msg.msg_flags = 0;
            let n = unsafe { libc::recvmsg(self.s.as_raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC) };
            if n < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            break n;
        };
        if n != 1 {
            return Err(os_err("rank link recvmsg closed"));
        }
        let mut fds = Vec::new();
        let mut invalid = msg.msg_flags & (libc::MSG_CTRUNC | libc::MSG_TRUNC) != 0;
        unsafe {
            let mut h = libc::CMSG_FIRSTHDR(&msg);
            while !h.is_null() {
                let header = libc::CMSG_LEN(0) as usize;
                let len = (*h).cmsg_len as usize;
                if len < header {
                    invalid = true;
                    break;
                }
                if (*h).cmsg_level == libc::SOL_SOCKET && (*h).cmsg_type == libc::SCM_RIGHTS {
                    let bytes = len - header;
                    invalid |= bytes % std::mem::size_of::<RawFd>() != 0;
                    for i in 0..bytes / std::mem::size_of::<RawFd>() {
                        let fd =
                            std::ptr::read_unaligned(libc::CMSG_DATA(h).cast::<RawFd>().add(i));
                        fds.push(OwnedFd::from_raw_fd(fd));
                    }
                } else {
                    invalid = true;
                }
                h = libc::CMSG_NXTHDR(&msg, h);
            }
        }
        if invalid || marker[0] > 1 || fds.len() != marker[0] as usize {
            return Err(RuntimeError::Device(
                "rank link invalid/truncated fd frame".into(),
            ));
        }
        let v = self.recv()?;
        Ok((v, fds.pop()))
    }
    pub fn send<T: serde::Serialize>(&mut self, v: &T) -> Result<()> {
        let b = serde_json::to_vec(v).map_err(|e| RuntimeError::Device(e.to_string()))?;
        self.s
            .write_all(&(b.len() as u32).to_le_bytes())
            .and_then(|_| self.s.write_all(&b))
            .map_err(|e| RuntimeError::Device(format!("rank link send: {e}")))
    }
    pub fn recv<T: serde::de::DeserializeOwned>(&mut self) -> Result<T> {
        let mut n = [0u8; 4];
        self.s
            .read_exact(&mut n)
            .map_err(|e| RuntimeError::Device(format!("rank link closed: {e}")))?;
        let mut b = vec![0u8; u32::from_le_bytes(n) as usize];
        self.s
            .read_exact(&mut b)
            .map_err(|e| RuntimeError::Device(format!("rank link closed: {e}")))?;
        serde_json::from_slice(&b).map_err(|e| RuntimeError::Device(e.to_string()))
    }
}

/// Clear close-on-exec so a spawned worker inherits `fd` under the same number.
pub fn inheritable(fd: RawFd) -> Result<()> {
    // SAFETY: fcntl on a descriptor the caller owns.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) } != 0 {
        return Err(os_err("fcntl FD_CLOEXEC"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsFd;

    #[test]
    fn fd_frames_preserve_json_boundaries_and_cloexec() {
        let (a, b) = UnixStream::pair().unwrap();
        let (mut a, mut b) = (Link::new(a), Link::new(b));
        let file = std::fs::File::open("/dev/null").unwrap();
        a.send(&"before").unwrap();
        a.send_fd(&(7u32, 4096u64), Some(file.as_fd())).unwrap();
        a.send_fd(&"unavailable", None).unwrap();
        a.send(&"after").unwrap();
        assert_eq!(b.recv::<String>().unwrap(), "before");
        let (metadata, fd) = b.recv_fd::<(u32, u64)>().unwrap();
        assert_eq!(metadata, (7, 4096));
        let fd = fd.unwrap();
        assert_ne!(
            unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFD) } & libc::FD_CLOEXEC,
            0
        );
        assert_eq!(std::fs::File::from(fd).read(&mut [0u8; 1]).unwrap(), 0);
        let (metadata, fd) = b.recv_fd::<String>().unwrap();
        assert_eq!(metadata, "unavailable");
        assert!(fd.is_none());
        assert_eq!(b.recv::<String>().unwrap(), "after");
    }

    #[test]
    fn invalid_fd_counts_and_truncation_close_received_descriptors() {
        for count in [2usize, 32] {
            let (a, b) = UnixStream::pair().unwrap();
            let mut b = Link::new(b);
            let mut pipe = [-1; 2];
            assert_eq!(
                unsafe { libc::pipe2(pipe.as_mut_ptr(), libc::O_NONBLOCK | libc::O_CLOEXEC) },
                0
            );
            let (reader, writer) =
                unsafe { (OwnedFd::from_raw_fd(pipe[0]), OwnedFd::from_raw_fd(pipe[1])) };
            let mut marker = [1u8];
            let mut iov = libc::iovec {
                iov_base: marker.as_mut_ptr().cast(),
                iov_len: 1,
            };
            let mut control = [0usize; 32];
            let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
            msg.msg_iov = &mut iov;
            msg.msg_iovlen = 1;
            msg.msg_control = control.as_mut_ptr().cast();
            msg.msg_controllen = unsafe { libc::CMSG_SPACE((count * 4) as u32) } as usize;
            unsafe {
                let h = libc::CMSG_FIRSTHDR(&msg);
                (*h).cmsg_level = libc::SOL_SOCKET;
                (*h).cmsg_type = libc::SCM_RIGHTS;
                (*h).cmsg_len = libc::CMSG_LEN((count * 4) as u32) as usize;
                for i in 0..count {
                    std::ptr::write_unaligned(
                        libc::CMSG_DATA(h).cast::<RawFd>().add(i),
                        writer.as_raw_fd(),
                    );
                }
                assert_eq!(libc::sendmsg(a.as_raw_fd(), &msg, libc::MSG_NOSIGNAL), 1);
            }
            drop(writer);
            assert!(b.recv_fd::<u32>().is_err());
            // A concurrent spawn can retain CLOEXEC writers briefly before its exec.
            let mut poll = libc::pollfd {
                fd: reader.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            assert_eq!(unsafe { libc::poll(&mut poll, 1, 1000) }, 1);
            assert_ne!(poll.revents & libc::POLLHUP, 0);
            // EOF proves that neither the received rights nor truncated rights leaked.
            let mut byte = [0u8];
            assert_eq!(
                unsafe { libc::read(reader.as_raw_fd(), byte.as_mut_ptr().cast(), 1) },
                0
            );
        }
    }

    fn dup(fd: RawFd) -> OwnedFd {
        // SAFETY: duplicating a live descriptor.
        unsafe { OwnedFd::from_raw_fd(libc::dup(fd)) }
    }

    #[test]
    fn barrier_releases_all_ranks_repeatedly() {
        let layout = PeerLayout::new(768, 4, 4).unwrap();
        let g0 = ProcGroup::create(3, 3, Some(layout), 1 << 16).unwrap();
        let g1 = ProcGroup::attach(dup(g0.fd()), 1, Some(layout)).unwrap();
        let g2 = ProcGroup::attach(dup(g0.fd()), 2, Some(layout)).unwrap();
        std::thread::scope(|s| {
            for g in [&g1, &g2] {
                s.spawn(move || {
                    for _ in 0..200 {
                        g.arm(7, Duration::from_secs(10)).unwrap();
                    }
                });
            }
            for _ in 0..200 {
                g0.arm(7, Duration::from_secs(10)).unwrap();
            }
        });
        assert_eq!(
            g1.region_offset(1) - g1.region_offset(0),
            g0.region_offset(1) - g0.region_offset(0)
        );
    }

    #[test]
    fn barrier_times_out_and_aborts() {
        let g0 = ProcGroup::create(2, 2, None, 0).unwrap();
        let _g1 = ProcGroup::attach(dup(g0.fd()), 1, None).unwrap();
        let e = g0.barrier(Duration::from_millis(50)).unwrap_err();
        assert!(matches!(e, RuntimeError::CollectiveBail(_)));
        assert!(g0.aborted());
        assert!(_g1.barrier(Duration::from_secs(5)).is_err());
    }

    #[test]
    fn barrier_detects_exited_peer() {
        let g0 = ProcGroup::create(2, 2, None, 0).unwrap();
        // A pid that has exited: a reaped child.
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let pid = child.id();
        child.wait().unwrap();
        g0.ctrl().pid[1].store(pid, Ordering::Release);
        let e = g0.barrier(Duration::from_secs(30)).unwrap_err();
        assert!(e.to_string().contains("rank 1"), "{e}");
    }

    #[test]
    fn own_xstate_zeroing_stays_in_own_region() {
        let layout = PeerLayout::new(768, 1, 4).unwrap();
        let g0 = ProcGroup::create(2, 2, Some(layout), 0).unwrap();
        let g1 = ProcGroup::attach(dup(g0.fd()), 1, Some(layout)).unwrap();
        let (b, len) = g0.mapping();
        // SAFETY: test-owned mapping.
        unsafe { std::ptr::write_bytes(b.add(4096), 0xAB, len - 4096) };
        g1.zero_own_xstate();
        assert_eq!(g1.own_xline(0), 0);
        assert_ne!(g0.own_xline(0), 0);
    }

    #[test]
    fn arm_rejects_a_diverging_plan_on_every_rank() {
        let g0 = ProcGroup::create(2, 2, None, 0).unwrap();
        let g1 = ProcGroup::attach(dup(g0.fd()), 1, None).unwrap();
        let (a, b) = std::thread::scope(|s| {
            let h = s.spawn(|| g1.arm(2, Duration::from_secs(10)));
            (g0.arm(1, Duration::from_secs(10)), h.join().unwrap())
        });
        assert!(a.is_err() && b.is_err());
        assert!(
            g0.arm(1, Duration::from_secs(1)).is_err(),
            "a failed group stays failed"
        );
    }

    #[test]
    fn subgroups_barrier_independently() {
        let g0 = ProcGroup::create(4, 2, None, 0).unwrap();
        let g1 = ProcGroup::attach(dup(g0.fd()), 1, None).unwrap();
        let _g2 = ProcGroup::attach(dup(g0.fd()), 2, None).unwrap();
        let _g3 = ProcGroup::attach(dup(g0.fd()), 3, None).unwrap();
        std::thread::scope(|s| {
            s.spawn(|| g1.arm(5, Duration::from_secs(10)).unwrap());
            g0.arm(5, Duration::from_secs(10)).unwrap();
        });
    }

    #[test]
    fn link_round_trips() {
        let (a, b) = UnixStream::pair().unwrap();
        let (mut a, mut b) = (Link::new(a), Link::new(b));
        a.send(&(7u32, vec![1u32, 2, 3])).unwrap();
        let got: (u32, Vec<u32>) = b.recv().unwrap();
        assert_eq!(got, (7, vec![1, 2, 3]));
        drop(a);
        assert!(b.recv::<u32>().is_err());
    }
}
