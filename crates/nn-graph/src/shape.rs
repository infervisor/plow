//! Tensor shapes: ordered lists of symbolic [`Dim`]s.

use crate::dim::{Dim, SymbolTable};
use smallvec::SmallVec;
use std::fmt;
use std::ops::Index;

/// A tensor shape. Rank is the number of dims; each dim may be static or
/// symbolic.
#[derive(Clone, PartialEq, Eq)]
pub struct Shape {
    dims: SmallVec<[Dim; 4]>,
}

impl Shape {
    pub fn new(dims: impl IntoIterator<Item = Dim>) -> Shape {
        Shape {
            dims: dims.into_iter().collect(),
        }
    }

    pub fn scalar() -> Shape {
        Shape {
            dims: SmallVec::new(),
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }

    pub fn dim(&self, i: usize) -> &Dim {
        &self.dims[i]
    }

    /// Last dimension (the feature/hidden axis, by convention).
    pub fn last(&self) -> Option<&Dim> {
        self.dims.last()
    }

    /// Product of all dims (total element count), kept symbolic.
    pub fn numel(&self) -> Dim {
        self.dims.iter().fold(Dim::stat(1), |acc, d| acc.mul(d))
    }

    /// Substitute concrete values for bound symbols in every dim.
    pub fn substitute(&self, values: &std::collections::HashMap<crate::dim::SymId, i64>) -> Shape {
        Shape::new(self.dims.iter().map(|d| d.substitute(values)))
    }

    /// `true` if every dim is statically known.
    pub fn is_fully_static(&self) -> bool {
        self.dims.iter().all(|d| d.is_static())
    }

    /// Pretty-print using interned symbol names.
    pub fn display_with(&self, syms: &SymbolTable) -> String {
        let inner: Vec<String> = self.dims.iter().map(|d| syms.display_dim(d)).collect();
        format!("[{}]", inner.join(", "))
    }
}

impl Index<usize> for Shape {
    type Output = Dim;
    fn index(&self, i: usize) -> &Dim {
        &self.dims[i]
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape{}", self)
    }
}
