//! Typed units. Newtypes so a byte count can never be passed where a bandwidth
//! is expected. All are `const`-constructible so device specs are `const` data.

/// A size in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(pub u64);

impl Bytes {
    pub const fn kib(n: u64) -> Self {
        Bytes(n * 1024)
    }
    pub const fn mib(n: u64) -> Self {
        Bytes(n * 1024 * 1024)
    }
    pub const fn gib(n: u64) -> Self {
        Bytes(n * 1024 * 1024 * 1024)
    }
    /// Value in whole KiB (truncating).
    pub const fn as_kib(self) -> u64 {
        self.0 / 1024
    }
    /// Value in whole GiB (truncating).
    pub const fn as_gib(self) -> u64 {
        self.0 / (1024 * 1024 * 1024)
    }
}

/// Memory/interconnect bandwidth in gigabytes per second (10^9 B/s, the
/// convention vendors quote and the scheduler's `BandwidthTree` reasons over).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct GBps(pub f64);

/// A clock frequency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Hertz(pub u64);

impl Hertz {
    pub const fn mhz(n: u64) -> Self {
        Hertz(n * 1_000_000)
    }
    /// Construct from a value in MHz (e.g. `1980` → 1.980 GHz). Avoids float
    /// for a const clock.
    pub const fn from_mhz(n: u64) -> Self {
        Hertz(n * 1_000_000)
    }
    /// Deprecated alias for [`Self::from_mhz`].
    #[deprecated(note = "use `from_mhz` — the name `ghz_milli` is ambiguous")]
    pub const fn ghz_milli(milli: u64) -> Self {
        Self::from_mhz(milli)
    }
}
