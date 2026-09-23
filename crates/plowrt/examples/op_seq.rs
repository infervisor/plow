use packet::dev::DevOp;
use plowrt::asset::devblob::DevBlob;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let buf = std::fs::read(&a[0]).expect("read blob");
    let blob = DevBlob::parse_l2(&buf, false).expect("parse blob");
    let pi: usize = a[1].parse().unwrap();
    let (lo, hi): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    for (i, p) in blob.progs.iter().enumerate() {
        println!("prog[{i}] T={} insts={}", p.t, p.insts.len());
    }
    let p = &blob.progs[pi];
    for (k, d) in p.insts.iter().enumerate().skip(lo).take(hi - lo) {
        let n = DevOp::from_u16(d.op).map(|o| o.c_name()).unwrap_or("?");
        let t: Vec<String> = d
            .t
            .iter()
            .take(4)
            .map(|&t| blob.tensors.get(t as usize).map_or("-".into(), |x| x.name.clone()))
            .collect();
        let seg = p.stream.iter().find(|e| e.inst as usize == k).map_or(-1, |e| e.seg as i32);
        println!(
            "{k:4} seg={seg:<4} {n:<34} blocks={:<4} i={:?} t={}",
            d.blocks,
            &d.i[..4],
            t.join(" | ")
        );
    }
}
