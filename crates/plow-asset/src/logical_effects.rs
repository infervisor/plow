//! Packet-derived whole-tensor effects. Physical aliases, kernel bounds and
//! counter fence implementations are separate obligations, not established here.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
use serde_json::{json, Value};

use crate::program::{Packet, Program};

fn outputs(d: &DevInst64) -> Result<&'static [usize], String> {
    use DevOp::*;
    let op = DevOp::from_u16(d.op).ok_or("unknown kernel effects")?;
    let required = match op {
        Nop => 0,
        ZeroF32 => 1,
        RmsNorm | CastF32Bf16 => 2,
        Residual | Glu | NormResidual | Gemm | Gemv | QuantFp8Block128 => 3,
        AddNorm | NormResidualNorm | MlaBmmFp8 => 4,
        GemmFp8Block128 => 5,
        _ => return Err(format!("unaudited logical effects for {op:?}")),
    };
    if d.t[..required].contains(&TENSOR_NONE16) {
        return Err("missing required effect operand".into());
    }
    match op {
        Nop => Ok(&[]),
        Residual | Glu | NormResidual | Gemm | ZeroF32 | CastF32Bf16 => Ok(&[0]),
        AddNorm | NormResidualNorm => Ok(&[0, 1]),
        RmsNorm if d.t[3] == TENSOR_NONE16 && d.t[4] == TENSOR_NONE16 => Ok(&[0]),
        Gemv if d.i[3] == 0 => Ok(&[0]),
        QuantFp8Block128 => Ok(&[0, 2]),
        // The multi-output QB/split variants require their own footprint audit.
        GemmFp8Block128 if d.i[4] == 0 && d.t[5..].iter().all(|&h| h == TENSOR_NONE16) => Ok(&[0]),
        MlaBmmFp8 if d.i[4] == 0 => Ok(&[0]),
        op => Err(format!("unaudited logical effects for {op:?}")),
    }
}

fn coarse_edges(program: &Program<'_>) -> Result<BTreeSet<(usize, usize)>, String> {
    let n = program.insts.len();
    if n == 0 || program.n_counter as usize != n || program.l2_domains != 0 {
        return Err("logical effects require ordinary coarse counters".into());
    }
    let mut seen = BTreeSet::new();
    let mut dependencies = vec![None; n];
    for entry in program.stream {
        let pc = entry.inst as usize;
        let inst = program
            .insts
            .get(pc)
            .ok_or("effect instruction out of range")?;
        if entry.flags != 0
            || entry.slice >= u32::from(inst.blocks)
            || !seen.insert((pc, entry.slice))
        {
            return Err("fine, duplicate or out-of-range effect slice".into());
        }
        let succ = program
            .succs
            .get(entry.succ_ofs as usize..entry.succ_ofs as usize + entry.succ_len as usize)
            .ok_or("invalid effect successor range")?;
        if succ != [pc as u32] {
            return Err("effect counter is not complete op retirement".into());
        }
        let waits = program
            .waits
            .get(entry.wait_ofs as usize..entry.wait_ofs as usize + entry.wait_len as usize)
            .ok_or("invalid effect wait range")?;
        let mut deps = BTreeSet::new();
        for wait in waits {
            let source = wait.id as usize;
            if source >= pc
                || wait.threshold != u32::from(program.insts[source].blocks)
                || !deps.insert(source)
            {
                return Err("effect wait lacks full earlier-producer completion".into());
            }
        }
        match &dependencies[pc] {
            Some(prior) if *prior != deps => {
                return Err("effect slices have different dependencies".into())
            }
            _ => dependencies[pc] = Some(deps),
        }
    }
    let mut edges = BTreeSet::new();
    for (pc, inst) in program.insts.iter().enumerate() {
        if inst.blocks == 0 || !(0..u32::from(inst.blocks)).all(|slice| seen.contains(&(pc, slice)))
        {
            return Err("missing effect producer slices".into());
        }
        for source in dependencies[pc]
            .as_ref()
            .ok_or("missing effect instruction")?
        {
            edges.insert((*source, pc));
        }
    }
    // The loaded GQ must execute exactly the slices whose counters were reconstructed.
    let canonical: BTreeMap<_, _> = program
        .stream
        .iter()
        .map(|e| ((e.inst, e.slice), e))
        .collect();
    let mut queued = BTreeSet::new();
    for entry in program.gq_stream {
        let original = canonical
            .get(&(entry.inst, entry.slice))
            .ok_or("unknown effect GQ slice")?;
        if !queued.insert((entry.inst as usize, entry.slice))
            || entry.wait_ofs != original.wait_ofs
            || entry.wait_len != original.wait_len
            || entry.succ_ofs != original.succ_ofs
            || entry.succ_len != original.succ_len
            || entry.flags != original.flags
            || entry.seg != original.seg
        {
            return Err("effect GQ gates differ from static stream".into());
        }
    }
    if queued != seen {
        return Err("missing effect GQ slices".into());
    }
    Ok(edges)
}

/// Normalize verified full-operation wire counters to one completion per task.
pub fn coarse_protocol(program: &Program<'_>) -> Result<Value, String> {
    let edges = coarse_edges(program)?;
    let n = program.insts.len();
    let mut waits = vec![Vec::new(); n];
    for (source, target) in edges {
        waits[target].push(source);
    }
    Ok(json!({"waits":waits,"succs":(0..n).map(|i|vec![i]).collect::<Vec<_>>(),
        "resource":(0..n).collect::<Vec<_>>(),"stream_idx":vec![0;n],"threshold":{}}))
}

/// One task per complete instruction, plus entry/exit sentinels. Each tensor
/// handle has its own logical pool; no physical non-aliasing claim is made.
pub fn obligation(packet: &Packet<'_>, index: usize) -> Result<Value, String> {
    let program = packet.programs.get(index).ok_or("missing effect program")?;
    if packet.tp
        || program.role.is_packed_sibling()
        || program.role.is_token_batch_body()
        || program.role.is_rowsplit_sibling()
    {
        return Err("peer/packed logical effects not qualified".into());
    }
    let edges = coarse_edges(program)?;
    let count = program.insts.len();
    let exit = count + 1;
    let n = count + 2;
    let mut accesses = Vec::new();
    let mut users: BTreeMap<u16, BTreeMap<usize, bool>> = BTreeMap::new();
    for (pc, inst) in program.insts.iter().enumerate() {
        let writes = outputs(inst)?;
        if inst.op == DevOp::Nop as u16 {
            continue;
        }
        for &slot in writes {
            if inst.t[slot] == TENSOR_NONE16 {
                return Err("missing effect output".into());
            }
        }
        for (slot, &handle) in inst.t.iter().enumerate() {
            if handle == TENSOR_NONE16 {
                continue;
            }
            let tensor = packet
                .tensors
                .get(handle as usize)
                .ok_or("effect tensor out of range")?;
            if tensor.bytes == 0 || tensor.name.contains("@") {
                return Err("empty or aliased effect tensor".into());
            }
            let write = writes.contains(&slot);
            users
                .entry(handle)
                .or_default()
                .entry(pc + 1)
                .and_modify(|prior| *prior |= write)
                .or_insert(write);
        }
    }
    if users.is_empty() {
        return Err("empty logical effect scope".into());
    }
    let mut waits = vec![Vec::new(); n];
    for task in 1..=count {
        waits[task].push(0usize);
    }
    for &(source, target) in &edges {
        waits[target + 1].push(source + 1);
    }
    waits[exit] = (1..=count).collect();
    let mut graph = vec![Vec::new(); n];
    for (target, sources) in waits.iter().enumerate() {
        for &source in sources {
            graph[source].push(target);
        }
    }
    let mut needed = BTreeMap::<usize, BTreeSet<usize>>::new();
    let mut leases = Vec::new();
    let mut address_map = Vec::new();
    let mut logical_offset = 0u64;
    for (&handle, tasks) in &users {
        let tensor = &packet.tensors[handle as usize];
        leases.push(
            json!({"pool":handle,"allocation":handle,"generation":0,"owner":0,
            "offset":0,"size":tensor.bytes,"acquire":0,"retire":exit,"cancel":null}),
        );
        address_map.push(json!({"name":format!("tensor:{handle}"),"cls":"Persistent",
            "offset":logical_offset,"size":tensor.bytes,
            "writers":tasks.iter().filter_map(|(&task,&write)| write.then_some(task)).collect::<Vec<_>>(),
            "readers":tasks.keys().copied().collect::<Vec<_>>() }));
        logical_offset = logical_offset
            .checked_add(tensor.bytes)
            .ok_or("logical effect extent overflow")?;
        for (&task, &write) in tasks {
            accesses.push(
                json!({"task":task,"pool":handle,"allocation":handle,"generation":0,
                "owner":0,"offset":0,"size":tensor.bytes,"write":write}),
            );
            needed.entry(0).or_default().insert(task);
            needed.entry(task).or_default().insert(exit);
            for (&other, &other_write) in tasks.range((task + 1)..) {
                if write || other_write {
                    needed.entry(task).or_default().insert(other);
                }
            }
        }
    }
    needed.entry(0).or_default().insert(exit);
    let mut paths = Vec::new();
    for (source, targets) in needed {
        let mut parent = vec![usize::MAX; n];
        let mut queue = VecDeque::from([source]);
        parent[source] = source;
        while let Some(node) = queue.pop_front() {
            for &next in &graph[node] {
                if parent[next] == usize::MAX {
                    parent[next] = node;
                    queue.push_back(next);
                }
            }
        }
        for target in targets {
            if parent[target] == usize::MAX {
                continue;
            } // Lean rejects the uncovered conflict.
            let mut via = Vec::new();
            let mut node = parent[target];
            while node != source {
                via.push(node);
                node = parent[node];
            }
            via.reverse();
            paths.push(json!({"source":source,"target":target,"via":via}));
        }
    }
    let dependencies: Vec<_> = edges.iter().map(|&(s, t)| [s + 1, t + 1]).collect();
    Ok(json!({
        "logical_effect_scope": "whole_declared_tensors; distinct logical pools; entry/exit lifetime; kernel bounds, physical aliases, runtime retirement and fence implementation external",
        "task_graph":{"n":n,"edges":dependencies},
        "protocol":{"waits":waits,"succs":(0..n).map(|id|vec![id]).collect::<Vec<_>>(),
            "threshold":(0..n).map(|id|(id.to_string(),1)).collect::<BTreeMap<_,_>>(),
            "resource":(0..n).collect::<Vec<_>>(),"stream_idx":vec![0;n]},
        "dependency_paths":vec![Vec::<usize>::new();edges.len()],
        "address_map":address_map,"address_paths":paths,
        "memory_effects":{"schema":1,"leases":leases,"accesses":accesses,
            "fence_counters":(0..n).collect::<Vec<_>>()}
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::{Builder, Model};

    fn model() -> Model {
        let mut b = Builder::new(1);
        let x = b.tensor("x", 16);
        let y = b.tensor("y", 16);
        let a = b.emit(DevOp::Residual, vec![0], &[], |d| {
            d.t[..3].copy_from_slice(&[y, x, x]);
            d.i[0] = 8;
        });
        b.emit(DevOp::Residual, vec![0], &[a], |d| {
            d.t[..3].copy_from_slice(&[x, y, y]);
            d.i[0] = 8;
        });
        let p = b.finish();
        Model {
            n_cu: 1,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        }
    }

    #[test]
    fn derives_real_operands_counters_and_logical_lifetimes() {
        let model = model();
        crate::program::with_model(&model, |packet| {
            let value = obligation(packet, 0).unwrap();
            assert_eq!(value["task_graph"], json!({"n":4,"edges":[[1,2]]}));
            assert_eq!(
                value["memory_effects"]["leases"].as_array().unwrap().len(),
                2
            );
            assert_eq!(
                value["memory_effects"]["accesses"]
                    .as_array()
                    .unwrap()
                    .len(),
                4
            );
            assert_eq!(value["address_map"].as_array().unwrap().len(), 2);
            assert_eq!(value["protocol"]["waits"][2], json!([0, 1]));
        });
    }

    #[test]
    fn rejects_uncovered_kernels_aliases_and_partial_completion() {
        for mutation in 0..5 {
            let mut model = model();
            match mutation {
                0 => model.progs[0].insts[0].op = DevOp::KdaStateStep as u16,
                1 => model.tensors[0].name = "x@band16".into(),
                2 => model.progs[0].waits[0].threshold = 0,
                3 => model.progs[0].gq_stream[0].succ_len = 0,
                _ => model.progs[0].insts[0].t[0] = packet::dev::TENSOR_NONE,
            }
            crate::program::with_model(&model, |packet| {
                assert!(obligation(packet, 0).is_err(), "mutation {mutation}")
            });
        }
    }
}
