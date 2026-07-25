//! Run egglog on the lowered program and parse the extracted term into a
//! hash-consed `FusedGraph` (a small Vec-indexed DAG owned by this crate — the
//! fused-op vocabulary stays out of `nn_graph`).

#[derive(thiserror::Error, Debug)]
pub enum ExtractError {
    #[error("egglog error: {0}")]
    Egglog(String),
    #[error("egglog produced no extraction output")]
    NoOutput,
    #[error("could not parse extracted term: {0}")]
    Parse(String),
}

/// One node of the post-rewrite operator DAG.
#[derive(Clone, Debug)]
pub struct FNode {
    pub op: String,
    pub args: Vec<Arg>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Node(usize),
    Str(String),
    Int(i64),
    Float(String),
}

/// Hash-consed DAG extracted from egglog (shared subterms deduplicated).
#[derive(Default, Debug)]
pub struct FusedGraph {
    pub nodes: Vec<FNode>,
    pub root: usize,
}

impl FusedGraph {
    /// Operation nodes only (excludes `Input`/`Weight` leaves).
    pub fn op_count(&self) -> usize {
        self.nodes.iter().filter(|n| !is_leaf(&n.op)).count()
    }

    /// Number of fused nodes produced by the rewrite rules.
    pub fn fused_count(&self) -> usize {
        self.nodes.iter().filter(|n| is_fused(&n.op)).count()
    }

    pub fn contains(&self, op: &str) -> bool {
        self.nodes.iter().any(|n| n.op == op)
    }
}

fn is_leaf(op: &str) -> bool {
    op == "Input" || op == "Weight"
}

fn is_fused(op: &str) -> bool {
    matches!(
        op,
        "FusedNormLinear"
            | "FusedNormLinearBias"
            | "FusedLayerNormLinear"
            | "FusedLayerNormLinearBias"
            | "SwiGLU"
            | "FusedGroupNormAct"
            | "FusedAdaLN"
            | "FusedGatedResidual"
            | "FusedNormRope"
            | "FusedNormRopeScale"
            | "FusedResidualNorm"
            | "FusedResidualLayerNorm"
            | "FusedGroupNormActConv3d"
            | "FusedGroupNormActConv3dBias"
            | "FusedLinearAct"
            | "FusedLinearBiasAct"
            | "FusedEmbeddingScale"
    )
}

/// Build the full egglog program, run it, and parse the extracted term.
pub fn run(schema: &str, rules: &str, lets: &str, root: &str) -> Result<FusedGraph, ExtractError> {
    let program = format!("{schema}\n{rules}\n{lets}\n(run 100)\n(extract {root})\n");

    let mut egraph = egglog::EGraph::default();
    let msgs = egraph
        .parse_and_run_program(None, &program)
        .map_err(|e| ExtractError::Egglog(e.to_string()))?;

    // The `(extract …)` result is the last message that is an s-expression.
    let term = msgs
        .iter()
        .rev()
        .map(|m| m.to_string())
        .find(|m| m.trim_start().starts_with('('))
        .ok_or(ExtractError::NoOutput)?;

    parse(term.trim())
}

// --- s-expression parsing into a hash-consed DAG ----------------------------

fn parse(s: &str) -> Result<FusedGraph, ExtractError> {
    let toks = tokenize(s);
    let mut p = Parser {
        toks: &toks,
        pos: 0,
        g: FusedGraph::default(),
        intern: std::collections::HashMap::new(),
    };
    let root = match p.parse_expr()? {
        Arg::Node(i) => i,
        other => {
            return Err(ExtractError::Parse(format!(
                "root is not a node: {other:?}"
            )))
        }
    };
    let mut g = p.g;
    g.root = root;
    Ok(g)
}

#[derive(Debug)]
enum Tok {
    Open,
    Close,
    Atom(String),
    Str(String),
}

fn tokenize(s: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '(' => {
                out.push(Tok::Open);
                chars.next();
            }
            ')' => {
                out.push(Tok::Close);
                chars.next();
            }
            '"' => {
                chars.next();
                let mut buf = String::new();
                for d in chars.by_ref() {
                    if d == '"' {
                        break;
                    }
                    buf.push(d);
                }
                out.push(Tok::Str(buf));
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            _ => {
                let mut buf = String::new();
                while let Some(&d) = chars.peek() {
                    if d == '(' || d == ')' || d == '"' || d.is_whitespace() {
                        break;
                    }
                    buf.push(d);
                    chars.next();
                }
                out.push(Tok::Atom(buf));
            }
        }
    }
    out
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
    g: FusedGraph,
    intern: std::collections::HashMap<String, usize>,
}

impl Parser<'_> {
    fn parse_expr(&mut self) -> Result<Arg, ExtractError> {
        match self.toks.get(self.pos) {
            Some(Tok::Str(s)) => {
                self.pos += 1;
                Ok(Arg::Str(s.clone()))
            }
            Some(Tok::Atom(a)) => {
                self.pos += 1;
                Ok(atom_arg(a))
            }
            Some(Tok::Open) => {
                self.pos += 1; // consume '('
                let op = match self.toks.get(self.pos) {
                    Some(Tok::Atom(a)) => a.clone(),
                    other => {
                        return Err(ExtractError::Parse(format!(
                            "expected constructor, got {other:?}"
                        )))
                    }
                };
                self.pos += 1;
                let mut args = Vec::new();
                loop {
                    match self.toks.get(self.pos) {
                        Some(Tok::Close) => {
                            self.pos += 1;
                            break;
                        }
                        None => return Err(ExtractError::Parse("unexpected end of input".into())),
                        _ => args.push(self.parse_expr()?),
                    }
                }
                Ok(Arg::Node(self.intern(op, args)))
            }
            Some(Tok::Close) | None => Err(ExtractError::Parse("unexpected token".into())),
        }
    }

    fn intern(&mut self, op: String, args: Vec<Arg>) -> usize {
        let key = node_key(&op, &args);
        if let Some(&i) = self.intern.get(&key) {
            return i;
        }
        let i = self.g.nodes.len();
        self.g.nodes.push(FNode { op, args });
        self.intern.insert(key, i);
        i
    }
}

fn atom_arg(a: &str) -> Arg {
    if let Ok(n) = a.parse::<i64>() {
        return Arg::Int(n);
    }
    // anything else numeric-looking (has '.', 'e') is a float token.
    Arg::Float(a.to_string())
}

fn node_key(op: &str, args: &[Arg]) -> String {
    let mut k = String::from(op);
    for a in args {
        k.push('|');
        match a {
            Arg::Node(i) => k.push_str(&format!("#{i}")),
            Arg::Str(s) => k.push_str(&format!("s:{s}")),
            Arg::Int(n) => k.push_str(&format!("i:{n}")),
            Arg::Float(f) => k.push_str(&format!("f:{f}")),
        }
    }
    k
}
