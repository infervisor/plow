//! §H Host-side text pipeline: tokenization, sampling, structured decoding.
//! These run on a tokio blocking pool so they overlap device compute.

pub mod guided;
pub mod sample;
pub mod tokenizer;
