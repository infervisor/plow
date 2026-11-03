use std::collections::BTreeMap;
use std::os::fd::AsFd;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;
use crate::device::cuda::ipc::IpcRegion;
use crate::exec::gpu::p2p::Probe;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
enum Mode {
    Auto,
    Host,
    P2p,
}

impl Mode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "host" => Ok(Self::Host),
            "p2p" => Ok(Self::P2p),
            _ => Err(RuntimeError::Rejected(format!(
                "PLOW_TP_TRANSPORT={value}: expected auto, host or p2p"
            ))),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Report {
    reason: Option<String>,
    fatal: bool,
    rtt_us: f64,
}

impl Report {
    fn error(be: &CudaBackend, e: RuntimeError) -> Self {
        Self {
            reason: Some(e.to_string()),
            fatal: be.ipc_must_retain(),
            rtt_us: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Decision {
    p2p: bool,
    fatal: bool,
    attempted: bool,
    reason: String,
    rtt_us: f64,
}

fn decide(mode: Mode, reports: &[Report]) -> Decision {
    let failure = reports.iter().enumerate().find(|(_, r)| r.reason.is_some());
    let fatal = reports.iter().any(|r| r.fatal);
    Decision {
        p2p: mode != Mode::Host && failure.is_none() && !fatal,
        fatal: fatal || (mode == Mode::P2p && failure.is_some()),
        attempted: mode != Mode::Host,
        reason: if mode == Mode::Host {
            "disabled by host selection".into()
        } else if let Some((rank, r)) = failure {
            format!("rank {rank}: {}", r.reason.as_deref().unwrap())
        } else {
            "interpreter one-shot/two-shot, slot reuse, readback and flag ping passed".into()
        },
        rtt_us: reports.iter().fold(0.0f64, |a, r| a.max(r.rtt_us)),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RegionInfo {
    rank: u32,
    bytes: u64,
    alignment: u64,
    digest: u64,
    report: Report,
}

fn layout_digest(layout: PeerLayout, tp: u32) -> u64 {
    [
        u64::from(tp),
        u64::from(layout.hidden),
        u64::from(layout.max_tokens),
        u64::from(layout.n_xctr),
        layout.bytes(),
        layout.partial_off(1).expect("slot 1"),
        layout.xctr_off(),
        layout.xstate_bytes(),
        layout.xstatus_off(),
        layout.xr_tag_off(),
        layout.xr_tag_slot(),
    ]
    .into_iter()
    .fold(0xcbf2_9ce4_8422_2325, |h, w| {
        (h ^ w).wrapping_mul(0x0100_0000_01b3)
    })
}

fn validate(info: &RegionInfo, fd: bool, rank: u32, layout: PeerLayout, tp: u32) -> Result<()> {
    if info.rank != rank
        || info.digest != layout_digest(layout, tp)
        || fd != info.report.reason.is_none()
        || (fd
            && (info.bytes < layout.bytes()
                || info.alignment < 2 << 20
                || !info.alignment.is_power_of_two()
                || info.bytes % info.alignment != 0))
    {
        return Err(RuntimeError::Device(format!(
            "rank {rank}: invalid IPC identity, layout or fd count"
        )));
    }
    Ok(())
}

enum Peers<'a> {
    Leader(&'a mut [Worker]),
    Worker(&'a mut Link),
}

impl Peers<'_> {
    fn agreement(
        &mut self,
        mode: Mode,
        local: Report,
        final_phase: bool,
    ) -> Result<Option<Decision>> {
        match self {
            Self::Leader(workers) => {
                let mut reports = vec![local];
                for w in workers.iter_mut() {
                    reports.push(w.link.recv::<Report>()?);
                }
                let done = final_phase || reports.iter().any(|r| r.reason.is_some() || r.fatal);
                let decision = done.then(|| decide(mode, &reports));
                for w in workers.iter_mut() {
                    w.link.send(&decision)?;
                }
                Ok(decision)
            }
            Self::Worker(link) => {
                link.send(&local)?;
                link.recv()
            }
        }
    }
}

pub(super) fn leader(
    be: Arc<CudaBackend>,
    group: Arc<ProcGroup>,
    assets: &Path,
    workers: &mut [Worker],
) -> Result<(RankBind, Decision)> {
    let mode = Mode::parse(&crate::config::RuntimeConfig::process_get().nv.tp_transport)?;
    for w in workers.iter_mut() {
        w.link.send(&mode)?;
    }
    negotiate(be, group, assets, mode, Peers::Leader(workers))
}

pub(super) fn worker(
    be: Arc<CudaBackend>,
    group: Arc<ProcGroup>,
    assets: &Path,
    link: &mut Link,
) -> Result<(RankBind, Decision)> {
    let mode = link.recv::<Mode>()?;
    negotiate(be, group, assets, mode, Peers::Worker(link))
}

fn negotiate(
    be: Arc<CudaBackend>,
    group: Arc<ProcGroup>,
    assets: &Path,
    mode: Mode,
    mut peers: Peers<'_>,
) -> Result<(RankBind, Decision)> {
    if mode == Mode::Host {
        return finish(be, group, Vec::new(), decide(mode, &[]));
    }
    let layout = group
        .layout()
        .ok_or_else(|| RuntimeError::Rejected("P2P needs collective layout".into()))?;
    let mut info = RegionInfo {
        rank: group.rank(),
        bytes: 0,
        alignment: 0,
        digest: layout_digest(layout, group.tp()),
        report: Report::default(),
    };
    let mut regions: Vec<Option<IpcRegion>> = (0..group.tp()).map(|_| None).collect();
    let own = match IpcRegion::create(be.clone(), layout.bytes()) {
        Ok((region, fd)) => {
            info.bytes = region.bytes;
            info.alignment = region.alignment;
            regions[group.rank() as usize] = Some(region);
            Some(fd)
        }
        Err(e) => {
            info.report = Report::error(&be, e);
            None
        }
    };
    let mut local = info.report.clone();
    match &mut peers {
        Peers::Leader(workers) => {
            let mut exports = vec![(info, own)];
            for (i, w) in workers.iter_mut().enumerate() {
                let (info, fd) = w.link.recv_fd::<RegionInfo>()?;
                validate(&info, fd.is_some(), i as u32 + 1, layout, group.tp())?;
                exports.push((info, fd));
            }
            let reports: Vec<_> = exports.iter().map(|(i, _)| i.report.clone()).collect();
            let decision = reports
                .iter()
                .any(|r| r.reason.is_some() || r.fatal)
                .then(|| decide(mode, &reports));
            for w in workers.iter_mut() {
                w.link.send(&decision)?;
            }
            if let Some(decision) = decision {
                return finish(be, group, regions.into_iter().flatten().collect(), decision);
            }
            // Send every descriptor before attempting local imports. A failed importer still
            // consumes its complete descriptor batch, leaving the JSON phase aligned.
            for (i, w) in workers.iter_mut().enumerate() {
                for (source, (info, fd)) in exports.iter().enumerate() {
                    if source != i + 1 {
                        w.link.send_fd(info, fd.as_ref().map(|f| f.as_fd()))?;
                    }
                }
            }
            for (source, (info, fd)) in exports.iter().enumerate().skip(1) {
                match IpcRegion::import(
                    be.clone(),
                    fd.as_ref().expect("validated fd").as_fd(),
                    info.bytes,
                    info.alignment,
                ) {
                    Ok(region) => regions[source] = Some(region),
                    Err(e) => merge_error(&mut local, Report::error(&be, e)),
                }
            }
        }
        Peers::Worker(link) => {
            link.send_fd(&info, own.as_ref().map(|f| f.as_fd()))?;
            if let Some(decision) = link.recv::<Option<Decision>>()? {
                return finish(be, group, regions.into_iter().flatten().collect(), decision);
            }
            for source in 0..group.tp() {
                if source == group.rank() {
                    continue;
                }
                let (info, fd) = link.recv_fd::<RegionInfo>()?;
                validate(&info, fd.is_some(), source, layout, group.tp())?;
                let fd = fd.ok_or_else(|| RuntimeError::Device("missing peer fd".into()))?;
                match IpcRegion::import(be.clone(), fd.as_fd(), info.bytes, info.alignment) {
                    Ok(region) => regions[source as usize] = Some(region),
                    Err(e) => merge_error(&mut local, Report::error(&be, e)),
                }
            }
        }
    }
    let mut regions: Vec<_> = regions.into_iter().flatten().collect();
    let mut probe = None;
    if local.reason.is_none() {
        match Probe::prepare(be.clone(), group.clone(), assets, &regions) {
            Ok(p) => probe = Some(p),
            Err(e) => local = Report::error(&be, e),
        }
    }
    if let Some(decision) = peers.agreement(mode, local, false)? {
        drop(probe);
        return finish(be, group, regions, decision);
    }
    let mut probe = probe.expect("all ranks prepared");
    let preliminary = match probe.addressability() {
        Ok((reason, rtt_us)) => Report {
            reason,
            fatal: false,
            rtt_us,
        },
        Err(e) => {
            for region in &mut regions {
                region.leak_on_drop();
            }
            Report {
                fatal: true,
                ..Report::error(&be, e)
            }
        }
    };
    let rtt_us = preliminary.rtt_us;
    if let Some(decision) = peers.agreement(mode, preliminary, false)? {
        drop(probe);
        return finish(be, group, regions, decision);
    }
    let mut report = match probe.run() {
        Ok((reason, rtt_us)) => Report {
            reason,
            fatal: false,
            rtt_us,
        },
        Err(e) => {
            for region in &mut regions {
                region.leak_on_drop();
            }
            Report {
                fatal: true,
                ..Report::error(&be, e)
            }
        }
    };
    report.rtt_us = rtt_us;
    drop(probe);
    if be.ipc_must_retain() {
        report.fatal = true;
        report
            .reason
            .get_or_insert_with(|| "probe stream failed to drain".into());
    }
    let decision = peers
        .agreement(mode, report, true)?
        .expect("final decision");
    finish(be, group, regions, decision)
}

fn merge_error(local: &mut Report, error: Report) {
    local.fatal |= error.fatal;
    if local.reason.is_none() {
        local.reason = error.reason;
    }
}

fn finish(
    be: Arc<CudaBackend>,
    group: Arc<ProcGroup>,
    regions: Vec<IpcRegion>,
    decision: Decision,
) -> Result<(RankBind, Decision)> {
    tracing::info!(rank = group.rank(), route = if decision.p2p { "p2p" } else { "host" },
        probe = %decision.reason, probe_round_trip_us = decision.rtt_us, fatal = decision.fatal,
        "tensor-parallel transport decision");
    if decision.fatal {
        group.abort();
        return Err(RuntimeError::Device(format!(
            "P2P bring-up failed: {}",
            decision.reason
        )));
    }
    let bind = if decision.p2p {
        RankBind::new_p2p(be, group, regions)?
    } else {
        drop(regions);
        RankBind::new(be, group)?
    };
    Ok((bind, decision))
}

static NEXT_METRIC: AtomicU64 = AtomicU64::new(0);
static METRICS: Mutex<BTreeMap<u64, (Decision, &'static str)>> = Mutex::new(BTreeMap::new());

pub(super) struct Metrics(u64);

impl Metrics {
    pub fn new(decision: Decision, tier: &'static str) -> Self {
        let id = NEXT_METRIC.fetch_add(1, Ordering::Relaxed);
        METRICS.lock().insert(id, (decision, tier));
        Self(id)
    }
}

impl Drop for Metrics {
    fn drop(&mut self) {
        METRICS.lock().remove(&self.0);
    }
}

pub(super) fn write_metrics(out: &mut String) {
    use std::fmt::Write;
    let metrics = METRICS.lock();
    if metrics.is_empty() {
        return;
    }
    out.push_str("# TYPE plow_tp_transport gauge\n# TYPE plow_tp_probe gauge\n# TYPE plow_tp_probe_round_trip_us gauge\n");
    for (id, (decision, tier)) in metrics.iter() {
        let route = if decision.p2p { "p2p" } else { "host" };
        let reason = crate::obs::serving::escape_label(&decision.reason);
        let _ = writeln!(
            out,
            "plow_tp_transport{{group=\"{id}\",route=\"{route}\",tier=\"{tier}\"}} 1"
        );
        let result = if decision.p2p {
            "pass"
        } else if decision.attempted {
            "fail"
        } else {
            "skipped"
        };
        let _ = writeln!(
            out,
            "plow_tp_probe{{group=\"{id}\",result=\"{result}\",reason=\"{reason}\"}} 1"
        );
        let _ = writeln!(
            out,
            "plow_tp_probe_round_trip_us{{group=\"{id}\"}} {}",
            decision.rtt_us
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_selection_is_group_wide_and_forced_p2p_is_strict() {
        let pass = Report::default();
        let fail = Report {
            reason: Some("flag timeout".into()),
            ..Report::default()
        };
        let fatal = Report {
            fatal: true,
            ..fail.clone()
        };
        for mode in [Mode::Auto, Mode::Host, Mode::P2p] {
            assert_eq!(
                decide(mode, &[pass.clone(), pass.clone()]).p2p,
                mode != Mode::Host
            );
            let fallback = decide(mode, &[pass.clone(), fail.clone()]);
            assert!(!fallback.p2p);
            assert_eq!(fallback.fatal, mode == Mode::P2p);
            assert!(decide(mode, &[fatal.clone()]).fatal);
        }
        assert!(Mode::parse("invalid").is_err());
    }

    #[test]
    fn descriptors_identify_ranks_and_layouts_not_device_ordinals() {
        let layout = PeerLayout::new(2560, 4096, 8).unwrap();
        let info = RegionInfo {
            rank: 1,
            bytes: 64 << 20,
            alignment: 2 << 20,
            digest: layout_digest(layout, 2),
            report: Report::default(),
        };
        assert!(validate(&info, true, 1, layout, 2).is_ok());
        assert!(validate(&info, true, 0, layout, 2).is_err());
        assert!(validate(&info, false, 1, layout, 2).is_err());
        let different = PeerLayout::new(2560, 4096, 9).unwrap();
        assert!(validate(&info, true, 1, different, 2).is_err());
    }
}
