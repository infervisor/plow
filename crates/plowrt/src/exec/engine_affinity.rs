//! Pin the AMD engine thread to the CPU socket that holds rank 0's GPU.
//!
//! The engine thread writes every AQL packet and kernarg, issues and waits on the SDMA counter
//! re-arms and polls the completion signals. With it on the socket away from rank 0's GPU all of
//! that, and the GPU's own dispatch retirement, run slower: a GLM-5.3 TP8 decode tick measured
//! 103.4 ms pinned to socket 1 against 97.5 ms pinned to rank 0's node, and unpinned processes
//! land on either socket at random (`host-shadowing.md` §5). A blocked drain wait recovers < 1 ms
//! of that, so the thread's placement is the fix, not the wait.

use std::path::Path;

/// `--amd-engine-affinity`: `auto` (rank 0's socket), `off`, or an explicit CPU list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineAffinity {
    Auto,
    Off,
    Cpus(Vec<usize>),
}

impl EngineAffinity {
    pub fn parse(spec: &str) -> Result<Self, String> {
        match spec.trim() {
            "" | "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            list => parse_cpulist(list)
                .filter(|cpus| !cpus.is_empty())
                .map(Self::Cpus)
                .ok_or_else(|| {
                    format!("--amd-engine-affinity `{spec}` is not auto, off or a CPU list like 0-23,192-215")
                }),
        }
    }
}

/// The kernel's cpulist format: `0-23,192-215` or `3`. `None` if malformed.
pub fn parse_cpulist(s: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in s.trim().split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((lo, hi)) => {
                let (lo, hi): (usize, usize) = (lo.parse().ok()?, hi.parse().ok()?);
                if hi < lo {
                    return None;
                }
                cpus.extend(lo..=hi);
            }
            None => cpus.push(part.parse().ok()?),
        }
    }
    Some(cpus)
}

/// The socket holding PCI device `bdf`: `(numa_node, package_id, online CPUs of that package)`,
/// read from `sysfs` (`/sys` in production). `None` when the device reports no NUMA node.
pub fn socket_cpus_of_pci(sysfs: &Path, bdf: &str) -> Option<(u32, u32, Vec<usize>)> {
    let read = |rel: &str| std::fs::read_to_string(sysfs.join(rel)).ok();
    let node: i64 = read(&format!("bus/pci/devices/{bdf}/numa_node"))?.trim().parse().ok()?;
    let node = u32::try_from(node).ok()?;
    let package = |cpu: usize| -> Option<u32> {
        read(&format!("devices/system/cpu/cpu{cpu}/topology/physical_package_id"))?
            .trim()
            .parse()
            .ok()
    };
    let first = *parse_cpulist(&read(&format!("devices/system/node/node{node}/cpulist"))?)?.first()?;
    let socket = package(first)?;
    let cpus: Vec<usize> = parse_cpulist(&read("devices/system/cpu/online")?)?
        .into_iter()
        .filter(|&cpu| package(cpu) == Some(socket))
        .collect();
    (!cpus.is_empty()).then_some((node, socket, cpus))
}

/// Restrict the calling thread to `cpus`.
pub fn pin_current_thread(cpus: &[usize]) -> std::io::Result<()> {
    // SAFETY: `cpu_set_t` is plain data, zero is the empty set, indices are bounded by
    // `CPU_SETSIZE`, and pid 0 names the calling thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &cpu in cpus.iter().filter(|&&c| c < libc::CPU_SETSIZE as usize) {
            libc::CPU_SET(cpu, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpulists_parse_like_the_kernel_prints_them() {
        assert_eq!(parse_cpulist("0-2,5"), Some(vec![0, 1, 2, 5]));
        assert_eq!(parse_cpulist(" 7 \n"), Some(vec![7]));
        assert_eq!(parse_cpulist(""), Some(vec![]));
        assert_eq!(parse_cpulist("3-1"), None);
        assert_eq!(parse_cpulist("a"), None);
        assert_eq!(EngineAffinity::parse("auto"), Ok(EngineAffinity::Auto));
        assert_eq!(EngineAffinity::parse("off"), Ok(EngineAffinity::Off));
        assert_eq!(EngineAffinity::parse("72-73,264"), Ok(EngineAffinity::Cpus(vec![72, 73, 264])));
        assert!(EngineAffinity::parse("near").is_err());
        assert!(EngineAffinity::parse(",").is_err());
    }

    /// A two-socket box shaped like the MI300X host: 4 NUMA nodes per socket, 2 CPUs per node
    /// plus their SMT siblings at +16, and a GPU on node 3 (socket 0). Auto must pick every
    /// online CPU of socket 0, siblings included, and nothing of socket 1.
    #[test]
    fn the_socket_comes_from_the_gpus_numa_node() {
        let root = std::env::temp_dir().join(format!("plow-affinity-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let write = |rel: &str, text: &str| {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        };
        write("bus/pci/devices/0000:05:00.0/numa_node", "3\n");
        write("bus/pci/devices/0000:85:00.0/numa_node", "7\n");
        write("bus/pci/devices/0000:99:00.0/numa_node", "-1\n");
        write("devices/system/cpu/online", "0-31\n");
        for node in 0..8usize {
            let cpus = format!("{}-{},{}-{}", 2 * node, 2 * node + 1, 16 + 2 * node, 17 + 2 * node);
            write(&format!("devices/system/node/node{node}/cpulist"), &cpus);
            for cpu in parse_cpulist(&cpus).unwrap() {
                write(
                    &format!("devices/system/cpu/cpu{cpu}/topology/physical_package_id"),
                    if node < 4 { "0\n" } else { "1\n" },
                );
            }
        }
        let (node, socket, cpus) = socket_cpus_of_pci(&root, "0000:05:00.0").unwrap();
        assert_eq!((node, socket), (3, 0));
        assert_eq!(cpus, [(0..8).collect::<Vec<_>>(), (16..24).collect()].concat());
        let (node, socket, cpus) = socket_cpus_of_pci(&root, "0000:85:00.0").unwrap();
        assert_eq!((node, socket), (7, 1));
        assert_eq!(cpus, [(8..16).collect::<Vec<_>>(), (24..32).collect()].concat());
        assert!(socket_cpus_of_pci(&root, "0000:99:00.0").is_none(), "no NUMA node, no pin");
        assert!(socket_cpus_of_pci(&root, "0000:aa:00.0").is_none(), "unknown device, no pin");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn pinning_the_calling_thread_takes_effect() {
        let cpu = std::thread::spawn(|| {
            pin_current_thread(&[0]).unwrap();
            // SAFETY: plain libc query of the calling thread's CPU.
            unsafe { libc::sched_getcpu() }
        })
        .join()
        .unwrap();
        assert_eq!(cpu, 0);
    }
}
