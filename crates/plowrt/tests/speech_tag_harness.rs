use std::collections::BTreeSet;

use plow_asset::speech_tags::SpeechTags;
use plowrt::tts::tags::{Parser, Span};

#[test]
fn fixture_canonical_tags_match_the_runtime_parser() {
    let cases: Vec<serde_json::Value> =
        serde_json::from_str(include_str!("../../../scripts/tts/tag_cases.json")).unwrap();
    let policy = SpeechTags::strip();
    for case in cases {
        let mut parser = Parser::default();
        parser.push(case["text"].as_str().unwrap(), &policy.syntax);
        let observed: BTreeSet<String> = parser
            .finish()
            .into_iter()
            .filter_map(|span| match span {
                Span::Tag { name, .. } if policy.tags.contains_key(&name) => Some(name),
                _ => None,
            })
            .collect();
        let expected: BTreeSet<String> =
            serde_json::from_value(case["canonical_tags"].clone()).unwrap();
        assert_eq!(observed, expected, "{}", case["id"]);
    }
}
