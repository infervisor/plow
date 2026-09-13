#[derive(Clone, Copy, Debug)]
pub struct LogMelSpec {
    pub sample_rate: u32,
    pub fft: u32,
    pub window: u32,
    pub hop: u32,
    pub bins: u32,
    pub preemphasis: f32,
    pub center_window: bool,
    pub periodic_hann: bool,
    pub normalize_per_feature: bool,
    pub mask_invalid_frames: bool,
    pub log_guard: f32,
    pub min_samples: u32,
    pub max_samples: u32,
}
