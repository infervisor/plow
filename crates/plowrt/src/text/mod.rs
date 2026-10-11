//! §H Host-side text pipeline: tokenization, sampling, structured decoding.
//! These run on a tokio blocking pool so they overlap device compute.

#[cfg(feature = "hf-tokenizer")]
mod detok;
pub mod guided;
pub mod logprobs;
pub mod rules;
pub mod sample;
pub mod segment;
pub mod tokenizer;
