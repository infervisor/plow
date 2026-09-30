//! §H Host-side text pipeline: tokenization, sampling, structured decoding.
//! These run on a tokio blocking pool so they overlap device compute.

pub mod engram;
pub mod guided;
pub mod logprobs;
pub mod sample;
pub mod rules;
pub mod segment;
pub mod tokenizer;
