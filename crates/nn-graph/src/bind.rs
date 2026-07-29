//! Binding symbolic size variables to concrete values.
//!
//! Shape inference ([`crate::infer_shapes`]) produces symbolic shapes (`[B, S,
//! 1000]`). To specialize a graph to a concrete batch / sequence / text length,
//! supply [`Bindings`] keyed by symbol name and call [`Graph::bind`] — every
//! inferred shape is rewritten with the bound symbols folded in. Unbound
//! symbols (e.g. binding only `B`) stay symbolic.
//!
//! Diffusion resolution is already static at build time (it changes the graph
//! structure, not just dim values), so binding is mainly for `B`, `S`, `L`.

use crate::dim::SymId;
use crate::graph::Graph;
use std::collections::HashMap;

/// Concrete values for symbolic size variables, keyed by name.
#[derive(Clone, Debug, Default)]
pub struct Bindings {
    values: HashMap<String, i64>,
}

impl Bindings {
    pub fn new() -> Self {
        Bindings::default()
    }

    /// Builder-style setter: `Bindings::new().set("B", 1).set("S", 512)`.
    pub fn set(mut self, name: &str, value: i64) -> Self {
        self.values.insert(name.to_string(), value);
        self
    }

    pub fn insert(&mut self, name: &str, value: i64) {
        self.values.insert(name.to_string(), value);
    }

    /// Resolve to id-keyed values against a graph's symbol table. Names not
    /// present in the graph are ignored.
    fn resolve(&self, graph: &Graph) -> HashMap<SymId, i64> {
        self.values
            .iter()
            .filter_map(|(name, &val)| graph.syms.id_of(name).map(|id| (id, val)))
            .collect()
    }
}

impl Graph {
    /// Specialize all inferred shapes by substituting bound symbols in place.
    /// Call after [`crate::infer_shapes`]. Idempotent for already-bound symbols.
    pub fn bind(&mut self, bindings: &Bindings) {
        let ids = bindings.resolve(self);
        if ids.is_empty() {
            return;
        }
        for t in &mut self.tensors {
            if let Some(shape) = &t.shape {
                t.shape = Some(shape.substitute(&ids));
            }
        }
    }
}
