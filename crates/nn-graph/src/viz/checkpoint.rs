use super::{CheckpointTensor, VizCheckpoint, VizNetwork};
use crate::Origin;
use std::collections::{BTreeMap, BTreeSet};

impl VizNetwork {
    /// Reconcile graph weights with checkpoint headers, without reading weight payloads.
    pub fn validate_checkpoint(
        &mut self,
        tensors: &BTreeMap<String, CheckpointTensor>,
    ) -> Result<(), String> {
        let mut matched = BTreeSet::new();
        for t in self
            .graph
            .tensors
            .iter()
            .filter(|t| matches!(t.origin, Origin::Weight))
        {
            if t.copies != 1 {
                return Err(format!(
                    "{}: aggregated expert {} cannot be shape-checked individually",
                    self.id, t.name
                ));
            }
            let actual = tensors
                .get(&t.name)
                .ok_or_else(|| format!("{}: checkpoint weight {} is missing", self.id, t.name))?;
            let expected: Vec<i64> = serde_json::from_str(&t.shape).map_err(|_| {
                format!(
                    "{}: weight {} has a non-static shape {}",
                    self.id, t.name, t.shape
                )
            })?;
            if expected != actual.shape {
                return Err(format!(
                    "{}: weight {} has checkpoint shape {:?}, graph expects {}",
                    self.id, t.name, actual.shape, t.shape
                ));
            }
            matched.insert(t.name.clone());
        }
        // Only update the graph once every shape has passed.
        for t in self
            .graph
            .tensors
            .iter_mut()
            .filter(|t| matches!(t.origin, Origin::Weight))
        {
            let actual = &tensors[&t.name];
            t.dtype = actual.dtype;
            t.bytes = Some(actual.bytes);
        }
        self.graph.summary.total_weight_bytes =
            matched.iter().map(|name| tensors[name].bytes).sum();
        for node in &mut self.graph.nodes {
            node.weight_bytes = node
                .inputs
                .iter()
                .map(|id| &self.graph.tensors[id.0 as usize])
                .filter(|t| matches!(t.origin, Origin::Weight))
                .map(|t| t.bytes.unwrap_or(0))
                .sum();
        }
        for block in &mut self.graph.blocks {
            block.weight_bytes = block
                .node_ids
                .iter()
                .map(|id| self.graph.nodes[id.0 as usize].weight_bytes)
                .sum();
        }
        self.checkpoint = Some(VizCheckpoint {
            tensor_count: tensors.len(),
            stored_bytes: tensors.values().map(|t| t.bytes).sum(),
            matched_tensors: matched.len(),
            unrepresented_tensors: tensors
                .keys()
                .filter(|name| !matched.contains(*name))
                .cloned()
                .collect(),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Dim, Nn};

    fn network() -> VizNetwork {
        let mut nn = Nn::new(DType::BF16, DType::BF16);
        let x = nn.input("x", nn.shape([Dim::stat(1), Dim::stat(4)]), DType::BF16);
        nn.begin_block("layer");
        let y = nn.linear("layer.proj", x, 4, 8, false);
        nn.end_block();
        nn.mark_output(y);
        let mut g = nn.finish();
        crate::infer_shapes(&mut g).unwrap();
        super::super::bundle_from_graph(&g, "t", "t", super::super::ModelModality::Llm)
            .networks
            .remove(0)
    }

    #[test]
    fn uses_checkpoint_dtype_and_storage_bytes() {
        let mut net = network();
        let tensors = BTreeMap::from([
            (
                "layer.proj.weight".into(),
                CheckpointTensor {
                    dtype: DType::F8E4M3,
                    shape: vec![8, 4],
                    bytes: 32,
                },
            ),
            (
                "layer.proj.weight_scale".into(),
                CheckpointTensor {
                    dtype: DType::F32,
                    shape: vec![8, 1],
                    bytes: 32,
                },
            ),
        ]);
        net.validate_checkpoint(&tensors).unwrap();
        assert_eq!(net.graph.summary.total_weight_bytes, 32);
        assert_eq!(net.graph.nodes[0].weight_bytes, 32);
        assert_eq!(net.graph.blocks[0].weight_bytes, 32);
        assert_eq!(net.checkpoint.as_ref().unwrap().stored_bytes, 64);
        assert_eq!(
            net.checkpoint.as_ref().unwrap().unrepresented_tensors,
            ["layer.proj.weight_scale"]
        );
    }

    #[test]
    fn missing_or_mismatched_weights_fail_without_mutating_graph() {
        let mut net = network();
        assert!(net
            .validate_checkpoint(&BTreeMap::new())
            .unwrap_err()
            .contains("missing"));
        let tensors = BTreeMap::from([(
            "layer.proj.weight".into(),
            CheckpointTensor {
                dtype: DType::F32,
                shape: vec![4, 8],
                bytes: 128,
            },
        )]);
        assert!(net
            .validate_checkpoint(&tensors)
            .unwrap_err()
            .contains("shape"));
        assert_eq!(net.graph.summary.total_weight_bytes, 64);
        assert!(net.checkpoint.is_none());
    }
}
