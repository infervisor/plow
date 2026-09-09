//! CPU topology from sysfs: which logical cpus are SMT siblings of one physical
//! core, and which NUMA node each belongs to. Pure parsers over the sysfs text so
//! tests feed fixtures; [`Topology::detect`] reads the live tree and falls back to
//! `available_parallelism` on one node when sysfs is absent (containers, non-Linux).

use std::path::Path;

/// One physical core: the representative logical cpu plus its SMT siblings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Core {
    /// Lowest-numbered logical cpu of the sibling set — the one a worker pins to.
    pub cpu: u32,
    pub node: u32,
    /// Every logical cpu sharing this core (includes `cpu`).
    pub siblings: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    /// Physical cores, ascending by `cpu`.
    pub cores: Vec<Core>,
    /// Online NUMA nodes, ascending.
    pub nodes: Vec<u32>,
}

/// Which NUMA nodes the pool spreads over (`--cpu-numa`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumaMode {
    /// Every allowed node, with interleaved large model allocations.
    Auto,
    /// Use all allowed CPUs without changing memory placement.
    Off,
    /// Restrict to these nodes.
    Nodes(Vec<u32>),
}

impl std::str::FromStr for NumaMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "auto" | "" => Ok(NumaMode::Auto),
            "off" | "none" => Ok(NumaMode::Off),
            list => {
                if list.split(',').any(|part| {
                    let mut ends = part.trim().split('-');
                    let a = ends.next().unwrap_or("").parse::<u32>();
                    let b = ends.next().map(str::parse::<u32>);
                    ends.next().is_some()
                        || match (a, b) {
                            (Ok(a), Some(Ok(b))) => a > b || b - a > 4096,
                            (Ok(_), None) => false,
                            _ => true,
                        }
                }) {
                    return Err(format!("invalid NUMA node list: {s:?}"));
                }
                let nodes = parse_cpulist(list);
                if nodes.is_empty() {
                    return Err(format!(
                        "--cpu-numa: expected auto|off|<node list>, got {s:?}"
                    ));
                }
                Ok(NumaMode::Nodes(nodes))
            }
        }
    }
}

/// Parse a sysfs cpulist (`0-3,8,10-11`) into sorted, deduplicated ids.
pub fn parse_cpulist(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
                    out.extend(a..=b);
                }
            }
            None => {
                if let Ok(v) = part.parse::<u32>() {
                    out.push(v);
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

impl Topology {
    /// Build from sysfs text: `online` cpulist, `(cpu, thread_siblings_list)` per
    /// online cpu, `(node, cpulist)` per node. Cpus missing from every node list
    /// land on node 0; cpus without a siblings entry are their own core.
    pub fn from_sysfs_text(
        online: &str,
        siblings: &[(u32, &str)],
        node_cpulists: &[(u32, &str)],
    ) -> Topology {
        let online = parse_cpulist(online);
        let node_of = |cpu: u32| -> u32 {
            node_cpulists
                .iter()
                .find(|(_, l)| parse_cpulist(l).contains(&cpu))
                .map(|(n, _)| *n)
                .unwrap_or(0)
        };
        let mut cores: Vec<Core> = Vec::new();
        for &cpu in &online {
            let mut sib = siblings
                .iter()
                .find(|(c, _)| *c == cpu)
                .map(|(_, l)| parse_cpulist(l))
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| vec![cpu]);
            sib.retain(|c| online.binary_search(c).is_ok());
            let rep = *sib.iter().min().unwrap_or(&cpu);
            if rep != cpu || cores.iter().any(|c| c.cpu == rep) {
                continue;
            }
            cores.push(Core {
                cpu,
                node: node_of(cpu),
                siblings: sib,
            });
        }
        let mut nodes: Vec<u32> = cores.iter().map(|c| c.node).collect();
        nodes.sort_unstable();
        nodes.dedup();
        Topology { cores, nodes }
    }

    /// Read the live sysfs tree; fall back to `available_parallelism` on node 0.
    pub fn detect() -> Topology {
        let root = Path::new("/sys/devices/system");
        let mut online = std::fs::read_to_string(root.join("cpu/online")).unwrap_or_default();
        #[cfg(target_os = "linux")]
        if let Ok(status) = std::fs::read_to_string("/proc/thread-self/status") {
            if let Some(allowed) = status
                .lines()
                .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
            {
                let allowed = parse_cpulist(allowed);
                online = parse_cpulist(&online)
                    .into_iter()
                    .filter(|c| allowed.binary_search(c).is_ok())
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
            }
        }
        if online.trim().is_empty() {
            return Topology::fallback();
        }
        let cpus = parse_cpulist(&online);
        let sib_text: Vec<(u32, String)> = cpus
            .iter()
            .filter_map(|&c| {
                let p = root.join(format!("cpu/cpu{c}/topology/thread_siblings_list"));
                std::fs::read_to_string(p).ok().map(|s| (c, s))
            })
            .collect();
        let mut node_text: Vec<(u32, String)> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(root.join("node")) {
            for e in rd.flatten() {
                let name = e.file_name();
                let name = name.to_string_lossy();
                if let Some(n) = name
                    .strip_prefix("node")
                    .and_then(|n| n.parse::<u32>().ok())
                {
                    if let Ok(l) = std::fs::read_to_string(e.path().join("cpulist")) {
                        if !l.trim().is_empty() {
                            node_text.push((n, l));
                        }
                    }
                }
            }
        }
        let sib: Vec<(u32, &str)> = sib_text.iter().map(|(c, s)| (*c, s.as_str())).collect();
        let nodes: Vec<(u32, &str)> = node_text.iter().map(|(n, s)| (*n, s.as_str())).collect();
        let t = Topology::from_sysfs_text(&online, &sib, &nodes);
        if t.cores.is_empty() {
            Topology::fallback()
        } else {
            t
        }
    }

    fn fallback() -> Topology {
        // macOS: only the performance cluster. Apple's E-cores have a fraction of the P-core
        // memory bandwidth and vector throughput, and the static per-CU streams hand every
        // worker the same share, so an E-core worker stalls the step (M4 Pro, Llama-3.2-3B bf16:
        // 8 P-core workers 43.8 ms/tok, 12 workers 105.6 ms/tok). No hard affinity exists on
        // Darwin; workers.rs asks for the user-interactive QoS class instead.
        #[cfg(all(target_os = "macos", feature = "cpu"))]
        let n = darwin_perf_cores().unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1) as u32
        });
        #[cfg(not(all(target_os = "macos", feature = "cpu")))]
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1) as u32;
        Topology {
            cores: (0..n)
                .map(|c| Core {
                    cpu: c,
                    node: 0,
                    siblings: vec![c],
                })
                .collect(),
            nodes: vec![0],
        }
    }

    pub fn physical_cores(&self) -> usize {
        self.cores.len()
    }

    pub fn worker_cpus(&self, nodes: &[u32]) -> Vec<(u32, u32)> {
        let groups: Vec<Vec<&Core>> = nodes
            .iter()
            .map(|&n| self.cores_on_node(n).collect())
            .collect();
        let ranks = self
            .cores
            .iter()
            .map(|c| c.siblings.len())
            .max()
            .unwrap_or(0);
        let width = groups.iter().map(Vec::len).max().unwrap_or(0);
        let mut cpus = Vec::new();
        for rank in 0..ranks {
            for i in 0..width {
                for cores in &groups {
                    if let Some(c) = cores.get(i) {
                        if let Some(&cpu) = c.siblings.get(rank) {
                            cpus.push((cpu, c.node));
                        }
                    }
                }
            }
        }
        cpus
    }

    /// Cores on `node`, ascending.
    pub fn cores_on_node(&self, node: u32) -> impl Iterator<Item = &Core> {
        self.cores.iter().filter(move |c| c.node == node)
    }

    /// The node set used for worker placement. Validate explicit nodes before loading.
    pub fn select_nodes(&self, mode: &NumaMode) -> Vec<u32> {
        match mode {
            NumaMode::Auto => self.nodes.clone(),
            NumaMode::Off => self.nodes.clone(),
            NumaMode::Nodes(list) => {
                let v: Vec<u32> = list
                    .iter()
                    .copied()
                    .filter(|n| self.nodes.contains(n))
                    .collect();
                if v.is_empty() {
                    vec![self.nodes.first().copied().unwrap_or(0)]
                } else {
                    v
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpulist_parses_ranges_and_singles() {
        assert_eq!(parse_cpulist("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpulist(""), Vec::<u32>::new());
        assert_eq!(parse_cpulist("5,5,1"), vec![1, 5]);
    }

    #[test]
    fn smt_siblings_collapse_to_one_core() {
        let t = Topology::from_sysfs_text(
            "0-3",
            &[(0, "0,2"), (1, "1,3"), (2, "0,2"), (3, "1,3")],
            &[(0, "0,2"), (1, "1,3")],
        );
        assert_eq!(t.physical_cores(), 2);
        assert_eq!(t.cores[0].cpu, 0);
        assert_eq!(t.cores[0].node, 0);
        assert_eq!(t.cores[1].cpu, 1);
        assert_eq!(t.cores[1].node, 1);
        assert_eq!(t.nodes, vec![0, 1]);
        assert_eq!(t.select_nodes(&NumaMode::Off), vec![0, 1]);
        assert_eq!(t.select_nodes(&NumaMode::Nodes(vec![7, 1])), vec![1]);
    }

    #[test]
    fn restricted_siblings_keep_allowed_representative() {
        let t = Topology::from_sysfs_text(
            "2-3",
            &[(2, "0,2"), (3, "1,3")],
            &[(0, "0,2"), (1, "1,3"), (2, "4-5")],
        );
        assert_eq!(
            t.cores.iter().map(|c| c.cpu).collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(t.cores[0].siblings, vec![2]);
        assert_eq!(t.nodes, vec![0, 1]);
    }

    #[test]
    fn placement_spreads_cores_before_smt() {
        let t = Topology::from_sysfs_text(
            "0-5",
            &[
                (0, "0,3"),
                (1, "1,4"),
                (2, "2,5"),
                (3, "0,3"),
                (4, "1,4"),
                (5, "2,5"),
            ],
            &[(0, "0-1,3-4"), (1, "2,5")],
        );
        assert_eq!(
            t.worker_cpus(&t.nodes),
            vec![(0, 0), (2, 1), (1, 0), (3, 0), (5, 1), (4, 0)]
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn detected_cpus_respect_thread_affinity() {
        let status = std::fs::read_to_string("/proc/thread-self/status").unwrap();
        let allowed = parse_cpulist(
            status
                .lines()
                .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
                .unwrap(),
        );
        let t = Topology::detect();
        let cpus = t.worker_cpus(&t.nodes);
        assert!(!cpus.is_empty());
        assert!(cpus.iter().all(|(c, _)| allowed.contains(c)));
        assert_eq!(cpus.len(), allowed.len());
    }

    #[test]
    fn numa_mode_parses() {
        assert_eq!("auto".parse::<NumaMode>().unwrap(), NumaMode::Auto);
        assert_eq!("off".parse::<NumaMode>().unwrap(), NumaMode::Off);
        assert_eq!(
            "0,2".parse::<NumaMode>().unwrap(),
            NumaMode::Nodes(vec![0, 2])
        );
        for bad in ["bogus", "0,bogus", "3-1", "0,", "-1", "0-4294967295"] {
            assert!(bad.parse::<NumaMode>().is_err(), "{bad}");
        }
    }
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
fn darwin_perf_cores() -> Option<u32> {
    let name = b"hw.perflevel0.logicalcpu\0";
    let mut v: u32 = 0;
    let mut len = std::mem::size_of::<u32>();
    // SAFETY: sysctlbyname with a NUL-terminated name, an out buffer of `len` bytes, no new value.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const libc::c_char,
            &mut v as *mut u32 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && v > 0).then_some(v)
}
