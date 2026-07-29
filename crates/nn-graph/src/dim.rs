//! Symbolic dimensions.
//!
//! A [`Dim`] is a multivariate polynomial over symbolic size variables (e.g.
//! batch `B`, sequence length `S`) with integer coefficients. This is richer
//! than the affine form sketched in the compiler design: flattening a
//! `[B, S, H]` activation to `[B*S, H]` produces the *product* `B*S`, which is
//! not affine. A polynomial closes under the operations shape inference needs —
//! add, subtract, multiply, and divide-by-static — while still canonicalizing
//! so that structurally equal shapes compare equal.
//!
//! Symbolic dims never appear in denominators in real transformer/vision
//! graphs (head_dim = hidden/heads is static/static), so division is only
//! defined where it stays exact.

use smallvec::{smallvec, SmallVec};
use std::fmt;

/// Interned identifier for a symbolic size variable.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SymId(pub u32);

/// A single monomial: `coeff * prod(var^power)`. Vars are kept sorted by
/// [`SymId`] so monomials have a canonical form.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Monomial {
    coeff: i64,
    vars: SmallVec<[(SymId, u32); 2]>,
}

impl Monomial {
    fn constant(c: i64) -> Self {
        Monomial {
            coeff: c,
            vars: SmallVec::new(),
        }
    }

    fn is_constant(&self) -> bool {
        self.vars.is_empty()
    }

    /// Multiply two monomials, merging matching variables and summing powers.
    fn mul(&self, other: &Monomial) -> Monomial {
        let mut vars: SmallVec<[(SymId, u32); 2]> = self.vars.clone();
        for &(id, p) in &other.vars {
            match vars.iter_mut().find(|(v, _)| *v == id) {
                Some(slot) => slot.1 += p,
                None => vars.push((id, p)),
            }
        }
        vars.sort_unstable_by_key(|(id, _)| id.0);
        Monomial {
            coeff: self.coeff * other.coeff,
            vars,
        }
    }

    /// Key used to group like terms (variable structure, ignoring coeff).
    fn var_key(&self) -> &[(SymId, u32)] {
        &self.vars
    }
}

/// A symbolic dimension: a canonicalized polynomial in size variables.
#[derive(Clone, PartialEq, Eq)]
pub struct Dim {
    /// Sorted, like-terms-combined, zero-coeff-free list of monomials.
    /// An empty vector represents the constant `0`.
    monomials: SmallVec<[Monomial; 1]>,
}

impl Dim {
    /// A concrete, statically-known size.
    pub fn stat(n: i64) -> Dim {
        if n == 0 {
            Dim {
                monomials: SmallVec::new(),
            }
        } else {
            Dim {
                monomials: smallvec![Monomial::constant(n)],
            }
        }
    }

    /// A bare symbolic variable (e.g. batch size `B`).
    pub fn sym(id: SymId) -> Dim {
        Dim {
            monomials: smallvec![Monomial {
                coeff: 1,
                vars: smallvec![(id, 1)]
            }],
        }
    }

    /// `true` if this dimension is a concrete integer; returns it if so.
    pub fn as_static(&self) -> Option<i64> {
        match self.monomials.as_slice() {
            [] => Some(0),
            [m] if m.is_constant() => Some(m.coeff),
            _ => None,
        }
    }

    pub fn is_static(&self) -> bool {
        self.as_static().is_some()
    }

    /// Collapse to canonical form: combine like terms, drop zero coeffs, sort.
    fn canonicalize(mut terms: Vec<Monomial>) -> Dim {
        // Sort by variable structure so like terms are adjacent.
        terms.sort_by(|a, b| a.var_key().cmp(b.var_key()));
        let mut out: SmallVec<[Monomial; 1]> = SmallVec::new();
        for m in terms {
            match out.last_mut() {
                Some(last) if last.var_key() == m.var_key() => last.coeff += m.coeff,
                _ => out.push(m),
            }
        }
        out.retain(|m| m.coeff != 0);
        Dim { monomials: out }
    }

    pub fn add(&self, other: &Dim) -> Dim {
        let mut terms = self.monomials.to_vec();
        terms.extend(other.monomials.iter().cloned());
        Dim::canonicalize(terms)
    }

    pub fn sub(&self, other: &Dim) -> Dim {
        let mut terms = self.monomials.to_vec();
        terms.extend(other.monomials.iter().map(|m| Monomial {
            coeff: -m.coeff,
            vars: m.vars.clone(),
        }));
        Dim::canonicalize(terms)
    }

    pub fn mul(&self, other: &Dim) -> Dim {
        let mut terms = Vec::with_capacity(self.monomials.len() * other.monomials.len());
        for a in &self.monomials {
            for b in &other.monomials {
                terms.push(a.mul(b));
            }
        }
        Dim::canonicalize(terms)
    }

    /// Substitute concrete values for some symbols, folding them into the
    /// coefficients. Unbound symbols are left in place, so the result may still
    /// be symbolic (e.g. binding `B` but not `S`).
    pub fn substitute(&self, values: &std::collections::HashMap<SymId, i64>) -> Dim {
        let mut terms: Vec<Monomial> = Vec::with_capacity(self.monomials.len());
        for m in &self.monomials {
            let mut coeff = m.coeff;
            let mut vars: SmallVec<[(SymId, u32); 2]> = SmallVec::new();
            for &(id, p) in &m.vars {
                match values.get(&id) {
                    Some(&val) => coeff *= val.pow(p),
                    None => vars.push((id, p)),
                }
            }
            terms.push(Monomial { coeff, vars });
        }
        Dim::canonicalize(terms)
    }

    /// Evaluate to a concrete integer given symbol values; `None` if any
    /// referenced symbol is unbound.
    pub fn eval(&self, values: &std::collections::HashMap<SymId, i64>) -> Option<i64> {
        self.substitute(values).as_static()
    }

    /// `true` if the two dims are provably unequal for every positive
    /// assignment of the symbols. Sound but incomplete: the difference must
    /// be a nonzero polynomial whose coefficients all share one sign (each
    /// monomial evaluates positive, so the difference cannot cancel to zero).
    /// `B*256` vs `B*512` and `S` vs `S+128` are provably unequal; `S` vs `L`
    /// or `B*256` vs `65536` are not (equal under some binding).
    pub fn provably_ne(&self, other: &Dim) -> bool {
        let d = self.sub(other);
        if d.monomials.is_empty() {
            return false; // identical polynomials
        }
        d.monomials.iter().all(|m| m.coeff > 0) || d.monomials.iter().all(|m| m.coeff < 0)
    }

    /// Exact division by a static integer. Returns `None` if it does not divide
    /// every coefficient evenly (we never introduce rationals into shapes).
    pub fn div_static(&self, d: i64) -> Option<Dim> {
        if d == 0 {
            return None;
        }
        let mut terms = Vec::with_capacity(self.monomials.len());
        for m in &self.monomials {
            if m.coeff % d != 0 {
                return None;
            }
            terms.push(Monomial {
                coeff: m.coeff / d,
                vars: m.vars.clone(),
            });
        }
        Some(Dim::canonicalize(terms))
    }
}

impl From<i64> for Dim {
    fn from(n: i64) -> Dim {
        Dim::stat(n)
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.monomials.is_empty() {
            return write!(f, "0");
        }
        for (i, m) in self.monomials.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            if m.is_constant() {
                write!(f, "{}", m.coeff)?;
                continue;
            }
            if m.coeff != 1 {
                write!(f, "{}*", m.coeff)?;
            }
            for (j, (id, p)) in m.vars.iter().enumerate() {
                if j > 0 {
                    write!(f, "*")?;
                }
                write!(f, "s{}", id.0)?;
                if *p != 1 {
                    write!(f, "^{}", p)?;
                }
            }
        }
        Ok(())
    }
}

impl fmt::Debug for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dim({})", self)
    }
}

/// Interns symbolic-variable names so graphs and pretty-printers can map
/// [`SymId`] back to readable names like `B` or `S`.
#[derive(Default, Debug, Clone)]
pub struct SymbolTable {
    names: Vec<String>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable::default()
    }

    /// Intern a name, returning a stable id. Re-interning the same name returns
    /// the same id.
    pub fn intern(&mut self, name: &str) -> SymId {
        if let Some(pos) = self.names.iter().position(|n| n == name) {
            return SymId(pos as u32);
        }
        let id = SymId(self.names.len() as u32);
        self.names.push(name.to_string());
        id
    }

    pub fn name(&self, id: SymId) -> Option<&str> {
        self.names.get(id.0 as usize).map(|s| s.as_str())
    }

    /// Look up the id for an already-interned name.
    pub fn id_of(&self, name: &str) -> Option<SymId> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|p| SymId(p as u32))
    }

    /// Render a dim using interned names instead of `s0`, `s1`, ...
    pub fn display_dim(&self, d: &Dim) -> String {
        let raw = d.to_string();
        // Replace from highest index to lowest to avoid substring collisions
        // (e.g. `s1` matching inside `s10`).
        let mut out = raw;
        for i in (0..self.names.len()).rev() {
            out = out.replace(&format!("s{}", i), &self.names[i]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_arithmetic() {
        assert_eq!(Dim::stat(4).add(&Dim::stat(6)).as_static(), Some(10));
        assert_eq!(Dim::stat(4).mul(&Dim::stat(6)).as_static(), Some(24));
        assert_eq!(Dim::stat(12).div_static(4).unwrap().as_static(), Some(3));
        assert!(Dim::stat(10).div_static(4).is_none());
    }

    #[test]
    fn symbolic_product_and_canonicalization() {
        let mut syms = SymbolTable::new();
        let b = Dim::sym(syms.intern("B"));
        let s = Dim::sym(syms.intern("S"));

        // B*S equals S*B (canonical ordering).
        assert_eq!(b.mul(&s), s.mul(&b));

        // (B*S) is not static.
        let bs = b.mul(&s);
        assert!(!bs.is_static());

        // B + B = 2B, then /2 = B.
        let two_b = b.add(&b);
        assert_eq!(two_b.div_static(2).unwrap(), b);

        // Display uses interned names.
        assert_eq!(syms.display_dim(&bs), "B*S");
    }

    #[test]
    fn affine_sequence_growth() {
        let mut syms = SymbolTable::new();
        let s = Dim::sym(syms.intern("S"));
        // S_kv = S + past, past static 128. (Constant term sorts first in the
        // canonical form.)
        let s_kv = s.add(&Dim::stat(128));
        assert_eq!(syms.display_dim(&s_kv), "128 + S");
        // (S+128) - 128 = S.
        assert_eq!(s_kv.sub(&Dim::stat(128)), s);
    }

    #[test]
    fn provably_unequal() {
        let mut syms = SymbolTable::new();
        let b = Dim::sym(syms.intern("B"));
        let s = Dim::sym(syms.intern("S"));
        let l = Dim::sym(syms.intern("L"));

        // Identical polynomials (canonicalized) are never provably unequal.
        assert!(!b.mul(&s).provably_ne(&s.mul(&b)));
        // Same variable, different coefficient: unequal for all B >= 1.
        assert!(b.mul(&Dim::stat(256)).provably_ne(&b.mul(&Dim::stat(512))));
        // S vs S+128: unequal.
        assert!(s.provably_ne(&s.add(&Dim::stat(128))));
        // Static vs static.
        assert!(Dim::stat(4096).provably_ne(&Dim::stat(1024)));
        // Different symbols: could be equal under some binding — not provable.
        assert!(!s.provably_ne(&l));
        // Symbolic vs static: B*256 == 65536 when B = 256 — not provable.
        assert!(!b.mul(&Dim::stat(256)).provably_ne(&Dim::stat(65536)));
        // Mixed-sign difference: 2B+4 vs 4B+2 (equal at B=1) — not provable.
        let a = b.mul(&Dim::stat(2)).add(&Dim::stat(4));
        let c = b.mul(&Dim::stat(4)).add(&Dim::stat(2));
        assert!(!a.provably_ne(&c));
    }
}
