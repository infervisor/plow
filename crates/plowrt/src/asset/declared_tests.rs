//! A version-2 packet is served from its declarations alone: the template is compiled with the
//! declared tool syntax and trace markers, a template-less packet gets exactly the builder it
//! names, and nothing is probed from the template text or the vocabulary.

use std::path::PathBuf;

use packet::devbuild::{Model, SectionData};
use plow_asset::serve_manifest::{self, ChatSpec, ReasoningDecl, ResponseSpec, ServeManifest};

use super::ModelBundle;
use crate::serve::chat::{chat_prompt, response_stream, PromptError};
use crate::serve::openai::{ChatRequest, Message};
use crate::serve::reasoning::ReasoningMode;
use crate::serve::template::RenderOpts;
use crate::serve::tools::request::{plan, Plan};
use crate::serve::tools::{ToolFormat, ToolSupport};

fn bundle(name: &str, m: &ServeManifest) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("plowrt-declared-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("weights.json"),
        r#"{"network":"declared-test","gpu":"test","num_gpus":1,"parallel":"none","weight_shared":false,"buckets":[]}"#,
    )
    .unwrap();
    let sections = vec![SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: serve_manifest::SECTION.into(),
        data: m.to_json(),
    }];
    let model = Model {
        n_cu: 1,
        target: 0,
        tensors: Vec::new(),
        progs: Vec::new(),
        kv_row_insts: Vec::new(),
        prog_t: Vec::new(),
        gen: Vec::new(),
    };
    std::fs::write(dir.join("model.pkt"), model.to_blob_v6(&sections)).unwrap();
    dir
}

fn v2(chat: ChatSpec) -> ServeManifest {
    ServeManifest {
        version: serve_manifest::VERSION,
        chat: Some(chat),
        ..Default::default()
    }
}

fn user(text: &str) -> Vec<Message> {
    vec![serde_json::from_value(serde_json::json!({"role": "user", "content": text})).unwrap()]
}

#[test]
fn a_declared_builder_is_used_without_probing_the_vocabulary() {
    let dir = bundle(
        "builtin",
        &v2(ChatSpec {
            builtin: Some("k3".into()),
            response: Some(ResponseSpec {
                tools: ResponseSpec::TOOLS_NONE.into(),
                reasoning: ReasoningDecl::None,
            }),
            ..Default::default()
        }),
    );
    let b = ModelBundle::load(&dir).expect("loads");
    assert!(b.chat_template().is_none());
    // The byte-fallback tokenizer here names no family; the probe would have fallen back to
    // Gemma's markers. The declaration decides.
    let p = chat_prompt(Some(&b), &user("hi"), &RenderOpts::default()).unwrap();
    assert!(
        p.starts_with("<|open|>message role=\"user\"<|sep|>hi"),
        "{p}"
    );
    assert_eq!(b.reasoning(), ReasoningMode::None);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn declared_tools_and_markers_drive_the_template_not_its_text() {
    // The template never renders `tools` and shows only an opening `<think>`: the probes would
    // say "no tools" and "no markers". The packet says otherwise, and the packet wins.
    let dir = bundle(
        "declared",
        &v2(ChatSpec {
            template: Some("{{ messages[0].content }}<think>".into()),
            response: Some(ResponseSpec {
                tools: ToolFormat::Hermes.name().into(),
                reasoning: ReasoningDecl::Tags {
                    open: "<think>".into(),
                    close: "</think>".into(),
                },
            }),
            ..Default::default()
        }),
    );
    let b = ModelBundle::load(&dir).expect("loads");
    let t = b.chat_template().expect("template");
    assert_eq!(t.tools, ToolSupport::Format(ToolFormat::Hermes));
    assert_eq!(t.reasoning, Some(("<think>", "</think>")));
    assert_eq!(b.reasoning(), ReasoningMode::ThinkTag);
    let p = chat_prompt(Some(&b), &user("q"), &RenderOpts::default()).unwrap();
    assert!(b.reasoning().prompt_opens(&p));
    std::fs::remove_dir_all(dir).unwrap();

    // And the other way round: a template full of `<think>` with a `none` declaration splits
    // nothing.
    let dir = bundle(
        "declared-none",
        &v2(ChatSpec {
            template: Some("{{ messages[0].content }}<think></think><think>".into()),
            response: Some(ResponseSpec {
                tools: ResponseSpec::TOOLS_NONE.into(),
                reasoning: ReasoningDecl::None,
            }),
            ..Default::default()
        }),
    );
    let b = ModelBundle::load(&dir).expect("loads");
    assert_eq!(b.chat_template().unwrap().tools, ToolSupport::None);
    assert_eq!(b.reasoning(), ReasoningMode::None);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_declared_none_overrides_a_template_that_renders_tools() {
    let dir = bundle(
        "tools-none",
        &v2(ChatSpec {
            template: Some(
                "{% for m in messages %}{{ m.content }}{% endfor %}{% if tools %}<tool_call>{{ tools | tojson }}{% endif %}"
                    .into(),
            ),
            response: Some(ResponseSpec {
                tools: ResponseSpec::TOOLS_NONE.into(),
                reasoning: ReasoningDecl::None,
            }),
            ..Default::default()
        }),
    );
    let b = ModelBundle::load(&dir).expect("loads");
    // The probe would have found Hermes; the packet says no tools, and the request is refused.
    assert_eq!(b.chat_template().unwrap().tools, ToolSupport::None);
    let req: ChatRequest = serde_json::from_value(serde_json::json!({"model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]}))
    .unwrap();
    let support = b.chat_template().map(|t| t.tools);
    assert!(plan(&req, support).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_declared_harmony_syntax_gets_its_parser_without_a_tool_request() {
    let with = |tools: &str, name: &str| {
        let dir = bundle(
            name,
            &v2(ChatSpec {
                template: Some("{{ messages[0].content }}".into()),
                response: Some(ResponseSpec {
                    tools: tools.into(),
                    reasoning: ReasoningDecl::None,
                }),
                ..Default::default()
            }),
        );
        let b = ModelBundle::load(&dir).expect("loads");
        let support = b.chat_template().map(|t| t.tools);
        let stream = response_stream(
            &Plan::default(),
            support,
            ReasoningMode::None,
            false,
            false,
            &b,
            false,
            None,
        )
        .is_some();
        std::fs::remove_dir_all(dir).unwrap();
        stream
    };
    assert!(with("harmony", "harmony-plain"));
    assert!(!with("none", "none-plain"));
}

/// Drive a generation through the response path a declared bundle gets, one push per piece as
/// the chat handler pushes deltas: the tool stream when `response_stream` builds one, else the
/// incremental reasoning split.
fn respond(b: &ModelBundle, prompt: &str, pieces: &[&str]) -> (String, String) {
    let mode = b.reasoning();
    let opens = mode.prompt_opens(prompt);
    let support = b.chat_template().map(|t| t.tools);
    let reenter = mode != ReasoningMode::None && b.think_markers().is_ok();
    match response_stream(
        &Plan::default(),
        support,
        mode,
        opens,
        reenter,
        b,
        true,
        None,
    ) {
        Some(mut ts) => {
            let (mut r, mut c) = (String::new(), String::new());
            let mut take = |st: crate::serve::tools::stream::Step| {
                r.push_str(st.reasoning.unwrap_or_default());
                c.push_str(st.content.unwrap_or_default());
            };
            for p in pieces {
                take(ts.push(p));
            }
            take(ts.finish(false));
            (r, c)
        }
        None => {
            // The streaming handler's path: one `ReasoningSplit` fed per generation delta.
            let mut split = crate::serve::reasoning::ReasoningSplit::new(mode, opens);
            let (mut r, mut c) = (String::new(), String::new());
            for p in pieces {
                split.push_into(p, &mut r, &mut c);
            }
            split.finish_into(&mut r, &mut c);
            (r, c)
        }
    }
}

#[test]
fn declared_syntaxes_drive_the_response_split() {
    // Harmony declared, no tool request: the channels are parsed, across a chunk boundary.
    let dir = bundle(
        "respond-harmony",
        &v2(ChatSpec {
            template: Some("{{ messages[0].content }}".into()),
            response: Some(ResponseSpec {
                tools: "harmony".into(),
                reasoning: ReasoningDecl::None,
            }),
            ..Default::default()
        }),
    );
    let b = ModelBundle::load(&dir).expect("loads");
    let prompt = chat_prompt(Some(&b), &user("q"), &RenderOpts::default()).unwrap();
    assert_eq!(
        respond(
            &b,
            &prompt,
            &[
                "<|channel|>analysis<|message|>Hm.<|end|><|start|>assis",
                "tant<|channel|>final<|message|>Four."
            ]
        ),
        ("Hm.".into(), "Four.".into())
    );
    std::fs::remove_dir_all(dir).unwrap();

    // `<think>` declared and left open by the template (Qwen): the close, split across two
    // pushes, ends the trace.
    let dir = bundle(
        "respond-think",
        &v2(ChatSpec {
            template: Some("{{ messages[0].content }}<think>\n".into()),
            response: Some(ResponseSpec {
                tools: ResponseSpec::TOOLS_NONE.into(),
                reasoning: ReasoningDecl::Tags {
                    open: "<think>".into(),
                    close: "</think>".into(),
                },
            }),
            ..Default::default()
        }),
    );
    let b = ModelBundle::load(&dir).expect("loads");
    let prompt = chat_prompt(Some(&b), &user("q"), &RenderOpts::default()).unwrap();
    assert!(b.reasoning().prompt_opens(&prompt));
    assert_eq!(
        respond(&b, &prompt, &["weighing it up</thi", "nk>Four."]),
        ("weighing it up".into(), "Four.".into())
    );
    std::fs::remove_dir_all(dir).unwrap();

    // A declared trace channel (Gemma 4's), opened by the template; the close split across pushes.
    let dir = bundle(
        "respond-channel",
        &v2(ChatSpec {
            template: Some("{{ messages[0].content }}<|channel>thought\n".into()),
            response: Some(ResponseSpec {
                tools: ResponseSpec::TOOLS_NONE.into(),
                reasoning: ReasoningDecl::Tags {
                    open: "<|channel>thought".into(),
                    close: "<channel|>".into(),
                },
            }),
            ..Default::default()
        }),
    );
    let b = ModelBundle::load(&dir).expect("loads");
    let prompt = chat_prompt(Some(&b), &user("q"), &RenderOpts::default()).unwrap();
    assert!(b.reasoning().prompt_opens(&prompt));
    assert_eq!(
        respond(&b, &prompt, &["plan<chan", "nel|>Answer"]),
        ("plan".into(), "Answer".into())
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn overlapping_markers_keep_the_version_1_precedence() {
    // A template with its own trace channel AND a tokenizer with the `<think>` pair: the
    // version-1 manifest declared `<think>` from the tokens and the runtime took the declaration
    // over the template; the version-2 emit declares the same.
    let dir = std::env::temp_dir().join(format!("plowrt-declared-overlap-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("chat_template.jinja"),
        "{{ messages[0].content }}<|channel>thought\n<channel|>",
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer.json"),
        r#"{"added_tokens":[{"content":"<think>"},{"content":"</think>"}]}"#,
    )
    .unwrap();
    let v2 = ServeManifest::emit(&dir, &dir).unwrap();
    let declared = ReasoningMode::declared(&v2.chat.unwrap().response.unwrap().reasoning);
    let probed = crate::serve::template::load(&dir).unwrap();
    let v1 = crate::asset::serve::ServeInfo {
        manifest: ServeManifest::from_checkpoint(&dir, &dir),
        multimodal: None,
        from_packet: true,
    };
    assert_eq!(declared, ReasoningMode::resolve(&v1, probed.reasoning));
    assert_eq!(declared, ReasoningMode::ThinkTag);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_version_2_template_that_does_not_compile_refuses_the_load() {
    let dir = bundle(
        "broken",
        &v2(ChatSpec {
            template: Some("{% if %}".into()),
            response: Some(ResponseSpec {
                tools: ResponseSpec::TOOLS_NONE.into(),
                reasoning: ReasoningDecl::None,
            }),
            ..Default::default()
        }),
    );
    let err = ModelBundle::load(&dir).err().expect("refused").to_string();
    assert!(err.contains("does not compile"), "{err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_version_1_packet_keeps_its_probes() {
    // Same template-less packet on version 1: no builder named, so the vocabulary probe runs
    // and, naming no family, yields Gemma's markers as before (the version-2 emit of such a
    // checkpoint declares no chat route instead: see `ServeManifest::emit`).
    let dir = bundle(
        "v1",
        &ServeManifest {
            version: serve_manifest::LEGACY_VERSION,
            chat: Some(ChatSpec::default()),
            ..Default::default()
        },
    );
    let b = ModelBundle::load(&dir).expect("loads");
    let p = chat_prompt(Some(&b), &user("hi"), &RenderOpts::default()).unwrap();
    assert!(p.contains("<|turn>user"), "{p}");
    std::fs::remove_dir_all(dir).unwrap();

    // A version-2 packet whose `chat` names neither is refused at parse, never probed; one
    // without a `chat` section serves no chat route at all.
    let bad = serde_json::json!({"version": 2, "chat": {
        "response": {"tools": "none", "reasoning": {"kind": "none"}}}});
    let err = ServeManifest::parse(bad.to_string().as_bytes()).unwrap_err();
    assert!(
        err.contains("neither a template nor a built-in builder"),
        "{err}"
    );
    let dir = bundle(
        "v2-no-chat",
        &ServeManifest {
            version: serve_manifest::VERSION,
            ..Default::default()
        },
    );
    let b = ModelBundle::load(&dir).expect("loads");
    assert!(matches!(
        chat_prompt(Some(&b), &user("hi"), &RenderOpts::default()),
        Err(PromptError::NoTemplate(_))
    ));
    std::fs::remove_dir_all(dir).unwrap();
}
