use egglog::ast::{Command, Expr, Literal, Parser};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term {
    Atom { kind: &'static str, value: String },
    Call { head: String, args: Vec<Term> },
}

fn term(expr: Expr) -> Term {
    match expr {
        Expr::Var(_, value) => Term::Atom {
            kind: "variable",
            value,
        },
        Expr::Call(_, head, args) => Term::Call {
            head,
            args: args.into_iter().map(term).collect(),
        },
        Expr::Lit(_, literal) => {
            let (kind, value) = match literal {
                Literal::String(s) => ("string", s),
                Literal::Int(n) => ("integer", n.to_string()),
                Literal::Float(n) => ("float_bits", n.0.to_bits().to_string()),
                Literal::Bool(b) => ("boolean", b.to_string()),
                Literal::Unit => ("unit", String::new()),
            };
            Term::Atom { kind, value }
        }
    }
}

/// Parse the actual engine syntax; conditional, bidirectional and subsuming
/// rewrites require distinct obligations and cannot enter this producer.
pub fn parse(source: &str) -> Result<Vec<(Term, Term)>, String> {
    Parser::default()
        .get_program_from_string(None, source)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|command| match command {
            Command::Rewrite(ruleset, rewrite, false)
                if ruleset.is_empty() && rewrite.conditions.is_empty() =>
            {
                Ok((term(rewrite.lhs), term(rewrite.rhs)))
            }
            _ => Err("rewrite-body proof requires ordinary unconditional one-way rules".into()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_rules_keep_all_operands_and_literal_kinds() {
        let bodies = parse(crate::rules_source()).unwrap();
        assert!(bodies.len() > 20);
        let (lhs, rhs) = &bodies[0];
        let Term::Call { head, args } = lhs else {
            panic!()
        };
        assert_eq!(head, "Linear");
        assert_eq!(args.len(), 3);
        let Term::Call { args, .. } = rhs else {
            panic!()
        };
        assert_eq!(args.len(), 5);
        assert_ne!(
            parse("(rewrite ?x \"?x\")").unwrap()[0].0,
            parse("(rewrite ?x \"?x\")").unwrap()[0].1
        );
        assert_ne!(
            parse("(rewrite 1 1.0)").unwrap()[0].0,
            parse("(rewrite 1 1.0)").unwrap()[0].1
        );
        for source in [
            "(birewrite ?x ?y)",
            "(rewrite ?x ?y :when ((= ?x ?y)))",
            "(rewrite ?x ?y :subsume)",
            "(let x 1)",
            "(rewrite",
        ] {
            assert!(parse(source).is_err(), "{source}");
        }
    }
}
