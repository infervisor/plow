W = '/root/plow/.claude/worktrees/tts-veena-chatterbox/crates/plowrt/src/'


def patch(path, pairs):
    s = open(W + path).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, s.count(a), a[:70])
        s = s.replace(a, b)
    open(W + path, 'w').write(s)


patch('tts/serving.rs', [
    ("""    if let Some(w) = workers().lock().get(&req.model).cloned() {
        return speech_on_worker(w, req, t_arrive).await;
    }""", """    // Bound first: a guard in the `if let` scrutinee would live across the await.
    let worker = workers().lock().get(&req.model).cloned();
    if let Some(w) = worker {
        return speech_on_worker(w, req, t_arrive).await;
    }"""),
])
patch('tts/chatterbox.rs', [
    ("                let mut pending: Vec<Option<Pending>> = Vec::new();",
     "                // Both callbacks touch the table; they never run at the same time.\n                let pending: std::cell::RefCell<Vec<Option<Pending>>> = Default::default();"),
    ("                        pending.push(Some(Pending {", "                        pending.borrow_mut().push(Some(Pending {"),
    ("                        let Some(p) = pending.get_mut(i).and_then(Option::take) else { return };",
     "                        let Some(p) = pending.borrow_mut().get_mut(i).and_then(Option::take) else { return };"),
])
print("ok")
