//! Probe: render a conversation through plowrt's ChatTemplate for a checkpoint dir.
//! argv[1] = dir, argv[2] = JSON array of messages. Prints JSON {source,out,err}.
use plowrt::serve::template::ChatTemplate;

fn main() {
    let dir = std::env::args().nth(1).expect("dir");
    let msgs_json = std::env::args().nth(2).expect("messages json");
    let msgs: Vec<serde_json::Value> = serde_json::from_str(&msgs_json).expect("messages");
    match ChatTemplate::load(std::path::Path::new(&dir)) {
        None => println!(
            "{}",
            serde_json::json!({"source": null, "out": null, "err": "NO TEMPLATE"})
        ),
        Some(t) => match t.render(&msgs) {
            Ok(out) => println!(
                "{}",
                serde_json::json!({"source": t.source, "out": out, "err": null})
            ),
            Err(e) => println!(
                "{}",
                serde_json::json!({"source": t.source, "out": null, "err": e})
            ),
        },
    }
}
