pub(super) fn kv_tensor_name(name: &str) -> Option<(u32, u32)> {
    let (layer, kind) = name.strip_prefix("kv.")?.split_once('.')?;
    let layer = layer.parse().ok()?;
    match kind {
        "k" => Some((layer, 0)),
        "v" => Some((layer, 1)),
        _ => None,
    }
}

pub(super) struct RingWindow {
    pub rows: u64,
    pub start: u64,
    pub first: u64,
}

impl RingWindow {
    // Callers validate the ring geometry when binding the packet.
    #[inline]
    pub fn new(frontier: u64, window: u64, ring: u64) -> Self {
        debug_assert!(ring.is_power_of_two() && window <= ring);
        let rows = frontier.min(window);
        let start = (frontier - rows) & (ring - 1);
        Self {
            rows,
            start,
            first: rows.min(ring - start),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_names_exclude_scales_and_non_kv_state() {
        assert_eq!(kv_tensor_name("kv.0.k"), Some((0, 0)));
        assert_eq!(kv_tensor_name("kv.4294967295.v"), Some((u32::MAX, 1)));
        for name in [
            "kv.1.ks",
            "kv.1.vs",
            "kv.1.ckv",
            "kv.1.k.extra",
            "kv.-1.k",
            "kv.4294967296.k",
            "act.0.k",
            "kv.k",
        ] {
            assert_eq!(kv_tensor_name(name), None, "{name}");
        }
    }

    #[test]
    fn ring_windows_cover_the_latest_rows_in_order() {
        for ring in [1, 2, 4, 16, 512] {
            for window in [0, 1, ring / 2, ring] {
                for frontier in 0..=ring * 3 {
                    let span = RingWindow::new(frontier, window, ring);
                    let got: Vec<_> = (span.start..span.start + span.first)
                        .chain(0..span.rows - span.first)
                        .collect();
                    let expected: Vec<_> = (frontier.saturating_sub(window)..frontier)
                        .map(|row| row % ring)
                        .collect();
                    assert_eq!(
                        got, expected,
                        "ring={ring} window={window} frontier={frontier}"
                    );
                }
            }
        }
    }
}
