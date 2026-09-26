p='/root/plow/.claude/worktrees/tts-veena-chatterbox/crates/plowrt/src/tts/t3.rs'
s=open(p).read()
a='assert_eq!(punc_norm("It was  a bright... day"), "It was a bright, day.");'
b='// Whitespace collapses BEFORE "..." -> ", ", so the reference keeps two spaces.\n        assert_eq!(punc_norm("It was  a bright... day"), "It was a bright,  day.");'
assert s.count(a)==1; s=s.replace(a,b)
a='''        // A draw never lands on a token below the min_p floor.
        for i in 0..1000 {
            let t = sample_cfg(&c, &cond, &uncond, &[1], Some(i as f32 / 1000.0), &mut s);
            assert!([1, 2, 5].contains(&t), "drew {t}");
        }'''
b='''        // penalty(tok 1) then /0.8: weights exp((x - 3)/0.8) = [.082, .535, 1, .0067, .0235, .829];
        // min_p 0.05 keeps {0, 1, 2, 5}, and every kept token is reachable.
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..1000 {
            let t = sample_cfg(&c, &cond, &uncond, &[1], Some(i as f32 / 1000.0), &mut s);
            assert!([0, 1, 2, 5].contains(&t), "drew {t}");
            seen.insert(t);
        }
        assert_eq!(seen.into_iter().collect::<Vec<_>>(), [0, 1, 2, 5]);'''
assert s.count(a)==1; s=s.replace(a,b)
open(p,'w').write(s); print('ok')
