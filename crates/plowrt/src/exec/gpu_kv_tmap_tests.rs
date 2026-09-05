use super::*;
use crate::asset::devblob::DevTensor;
use packet::rope::GenTensor;

fn fixture(batch: usize) -> (Vec<DevTensor>, GenTensor) {
    let tensors = [
        256,
        batch as u64 * 2 * 128 * 256 * 2,
        batch as u64 * 2 * 128 * 256 * 2,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, bytes)| DevTensor {
        name: format!("tensor{i}"),
        bytes,
        init: None,
    })
    .collect();
    (tensors, GenTensor::tmap_kv_pair(1, 2, 128, 256, 2))
}

#[test]
fn slot_tables_bind_matching_cache_addresses_and_immutable_descriptors() {
    for batch in [1, 2, 16] {
        let (tensors, recipe) = fixture(batch);
        let maps = kv_tensor_maps(&tensors, &[recipe], batch).unwrap();
        let map = &maps[0];
        let ptrs = [0x1000, 0x10000000, 0x20000000, 0x30000000];
        for slot in 0..batch {
            let descriptor = ptrs[0] + slot as u64 * 256;
            let bindings = map.slot_bindings(&ptrs, slot, descriptor).unwrap();
            let mut shifted = ptrs;
            for (id, address) in bindings {
                shifted[id] = address;
            }
            let offset = slot as u64 * 2 * 128 * 256 * 2;
            assert_eq!(
                shifted,
                [descriptor, ptrs[1] + offset, ptrs[2] + offset, ptrs[3]]
            );
            for i in 1..=2 {
                assert!(shifted[i] + map.stride <= ptrs[i] + tensors[i].bytes);
            }
            if slot == 0 {
                assert_eq!(shifted, ptrs);
            }
        }
        assert_eq!(ptrs, [0x1000, 0x10000000, 0x20000000, 0x30000000]);
        assert!(map.slot_bindings(&ptrs, batch, 0x1000).is_err());
    }
}

#[test]
fn rejects_invalid_pair_handles_and_extents() {
    let (tensors, recipe) = fixture(2);
    for bytes in [0, 128, 255, 257, 512] {
        let (mut bad, _) = fixture(2);
        bad[0].bytes = bytes;
        assert!(kv_tensor_maps(&bad, &[recipe], 2).is_err());
    }
    for i in [1, 2] {
        for bytes in [0, tensors[i].bytes - 1, tensors[i].bytes + 2] {
            let (mut bad, _) = fixture(2);
            bad[i].bytes = bytes;
            assert!(kv_tensor_maps(&bad, &[recipe], 2).is_err());
        }
    }
    for (map, k, v) in [
        (3, 1, 2),
        (0, 3, 2),
        (0, 1, 3),
        (0, 1, 1),
        (0, 0, 2),
        (0, 1, 0),
    ] {
        let bad = GenTensor {
            tensor: map,
            aux: k,
            scale: v,
            ..recipe
        };
        assert!(kv_tensor_maps(&tensors, &[bad], 2).is_err());
    }
    assert!(kv_tensor_maps(&tensors, &[recipe], 0).is_err());
    assert!(kv_tensor_maps(&tensors, &[recipe], 1).is_err());
    assert!(kv_tensor_maps(&tensors, &[recipe, recipe], 2).is_err());
    let generated_target = GenTensor {
        tensor: 1,
        ..recipe
    };
    assert!(kv_tensor_maps(&tensors, &[recipe, generated_target], 2).is_err());
}

#[test]
fn rejects_invalid_or_conflicting_geometry() {
    let (tensors, recipe) = fixture(2);
    for heads in [
        0.0,
        -1.0,
        1.5,
        f64::NAN,
        f64::INFINITY,
        u32::MAX as f64 + 1.0,
    ] {
        let bad = GenTensor {
            frac: heads,
            ..recipe
        };
        assert!(kv_tensor_maps(&tensors, &[bad], 2).is_err());
    }
    for (rows, hd, heads) in [
        (0, 256, 2.0),
        (128, 0, 2.0),
        (128, 255, 2.0),
        (64, 256, 2.0),
        (u32::MAX, u32::MAX - 63, u32::MAX as f64),
    ] {
        let bad = GenTensor {
            ctx: rows,
            hd,
            frac: heads,
            ..recipe
        };
        assert!(kv_tensor_maps(&tensors, &[bad], 2).is_err());
    }
    let mut shared = tensors;
    shared.push(DevTensor {
        name: "second_map".into(),
        bytes: 256,
        init: None,
    });
    let same_pair = GenTensor {
        tensor: 3,
        ..recipe
    };
    assert_eq!(
        kv_tensor_maps(&shared, &[recipe, same_pair], 2)
            .unwrap()
            .len(),
        2
    );
    let incompatible = GenTensor {
        tensor: 3,
        ctx: 64,
        frac: 4.0,
        ..recipe
    };
    assert!(kv_tensor_maps(&shared, &[recipe, incompatible], 2).is_err());
}

#[test]
fn rejects_misaligned_and_overflowing_slot_addresses() {
    let (tensors, recipe) = fixture(2);
    let maps = kv_tensor_maps(&tensors, &[recipe], 2).unwrap();
    let map = &maps[0];
    let ptrs = [0x1000, 0x10000000, 0x20000000];
    for descriptor in [0, 0x1001, 0x1040] {
        assert!(map.slot_bindings(&ptrs, 1, descriptor).is_err());
    }
    for target in [0, 0x10000001, u64::MAX - 15, u64::MAX - map.stride + 1] {
        for i in [1, 2] {
            let mut bad = ptrs;
            bad[i] = target;
            assert!(map.slot_bindings(&bad, 1, 0x1000).is_err());
        }
    }
    assert!(map.slot_bindings(&ptrs[..2], 1, 0x1000).is_err());
}

#[test]
#[ignore = "requires TEST_KV_TMAP_PACKET pointing to a real packet"]
fn actual_packet_slot_descriptors() {
    let path = std::env::var("TEST_KV_TMAP_PACKET").expect("packet path");
    let blob = DevBlob::parse(&std::fs::read(path).unwrap()).unwrap();
    let batch = blob.decode_prog().unwrap().t as usize;
    let maps = kv_tensor_maps(&blob.tensors, &blob.gen, batch).unwrap();
    assert!(!maps.is_empty());
    let mut cursor = 0x10000000u64;
    let ptrs: Vec<_> = blob
        .tensors
        .iter()
        .map(|tensor| {
            let base = cursor;
            cursor += tensor.bytes.div_ceil(128) * 128;
            base
        })
        .collect();
    for slot in 0..batch {
        for map in &maps {
            let descriptor = if slot == 0 { ptrs[map.tensor] } else { cursor };
            cursor += 256;
            let bindings = map.slot_bindings(&ptrs, slot, descriptor).unwrap();
            assert_eq!(bindings[0], (map.tensor, descriptor));
            for &(id, base) in &bindings[1..] {
                assert_eq!(
                    base,
                    ptrs[id] + slot as u64 * blob.tensors[id].bytes / batch as u64
                );
                assert!(base + map.stride <= ptrs[id] + blob.tensors[id].bytes);
            }
        }
    }
    println!(
        "validated {} descriptor pairs across {batch} slots",
        maps.len()
    );
}
