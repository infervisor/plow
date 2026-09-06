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
    /// Every online node, one queue/arena policy per node.
    Auto,
    /// Ignore topology: one node, no pinning by node.
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
                let nodes = parse_cpulist(list);
                if nodes.is_empty() {
                    return Err(format!("--cpu-numa: expected auto|off|<node list>, got {s:?}"));
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
            let sib = siblings
                .iter()
                .find(|(c, _)| *c == cpu)
                .map(|(_, l)| parse_cpulist(l))
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| vec![cpu]);
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
        let mut nodes: Vec<u32> = node_cpulists.iter().map(|(n, _)| *n).collect();
        if nodes.is_empty() {
            nodes.push(0);
        }
        nodes.sort_unstable();
        nodes.dedup();
        Topology { cores, nodes }
    }

    /// Read the live sysfs tree; fall back to `available_parallelism` on node 0.
    pub fn detect() -> Topology {
        let root = Path::new("/sys/devices/system");
        let online = std::fs::read_to_string(root.join("cpu/online")).unwrap_or_default();
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
                if let Some(n) = name.strip_prefix("node").and_then(|n| n.parse::<u32>().ok()) {
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

    /// Cores on `node`, ascending.
    pub fn cores_on_node(&self, node: u32) -> impl Iterator<Item = &Core> {
        self.cores.iter().filter(move |c| c.node == node)
    }

    /// The node set a [`NumaMode`] selects, in ascending order (never empty:
    /// `Off` collapses to `[0]`, `Nodes` keeps only nodes that exist).
    pub fn select_nodes(&self, mode: &NumaMode) -> Vec<u32> {
        match mode {
            NumaMode::Auto => self.nodes.clone(),
            NumaMode::Off => vec![self.nodes.first().copied().unwrap_or(0)],
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
        assert_eq!(t.select_nodes(&NumaMode::Off), vec![0]);
        assert_eq!(t.select_nodes(&NumaMode::Nodes(vec![7, 1])), vec![1]);
    }

    #[test]
    fn numa_mode_parses() {
        assert_eq!("auto".parse::<NumaMode>().unwrap(), NumaMode::Auto);
        assert_eq!("off".parse::<NumaMode>().unwrap(), NumaMode::Off);
        assert_eq!("0,2".parse::<NumaMode>().unwrap(), NumaMode::Nodes(vec![0, 2]));
        assert!("bogus".parse::<NumaMode>().is_err());
    }
}
