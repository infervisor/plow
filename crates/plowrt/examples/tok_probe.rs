//! Probe: does a checkpoint's tokenizer resolve its chat-template markers to
//! single ids? A marker that does not is spelled out as ordinary text, which is
//! what makes a correct-looking prompt a wrong one.
//! argv[1] = dir, argv[2..] = marker strings.
fn main() {
    let dir = std::env::args().nth(1).expect("dir");
    let tok = plowrt::text::tokenizer::load_tokenizer(std::path::Path::new(&dir));
    println!(
        "byte_fallback={} vocab_size={}",
        tok.is_byte_fallback(),
        tok.vocab_size()
    );
    for marker in std::env::args().skip(2) {
        let ids = tok.encode(&marker);
        let back = tok.decode(&ids);
        println!(
            "{:<24} ids={:?} single={} roundtrip={}",
            marker,
            ids,
            ids.len() == 1,
            back == marker
        );
    }
}
