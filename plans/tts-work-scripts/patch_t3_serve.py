p = '/root/plow/.claude/worktrees/tts-veena-chatterbox/crates/plowrt/src/tts/t3.rs'
s = open(p).read()
start = s.index("    /// Run every job to completion; `on_done(index, output)` fires as each finishes.")
end = s.index("#[cfg(test)]")
new = '''    /// Run every job to completion; `on_done(index, output)` fires as each finishes.
    pub fn run(&mut self, jobs: &[T3Job], on_done: impl FnMut(usize, T3Output)) -> Result<()> {
        let mut it = jobs.iter();
        self.serve(
            |_| {
                it.next().map(|j| T3Job {
                    voice: j.voice.clone(),
                    text: j.text.clone(),
                    seed: j.seed,
                    max_tokens: j.max_tokens,
                })
            },
            on_done,
        )
    }

    /// Continuous batching over slot pairs. `next(block)` supplies the next job (arrival index =
    /// call order); it is polled while a pair is free and asked to BLOCK only when nothing is in
    /// flight, and `None` from a blocking call ends the loop. `on_done(index, output)` fires as
    /// each job finishes. A job whose admission fails (unknown voice, text too long) is reported
    /// through `on_done` with no tokens and does not stop the loop.
    pub fn serve(
        &mut self,
        mut next: impl FnMut(bool) -> Option<T3Job>,
        mut on_done: impl FnMut(usize, T3Output),
    ) -> Result<()> {
        let mut jobs: Vec<T3Job> = Vec::new();
        let mut outputs: Vec<T3Output> = Vec::new();
        let mut started: Vec<std::time::Instant> = Vec::new();
        let mut active: Vec<Option<Active>> = (0..self.pairs()).map(|_| None).collect();
        let mut toks = Vec::new();
        let mut feeds = Vec::new();
        let mut closed = false;
        loop {
            for pair in 0..active.len() {
                if active[pair].is_some() || closed {
                    continue;
                }
                let idle = active.iter().all(Option::is_none);
                let Some(job) = next(idle) else {
                    closed = idle;
                    break;
                };
                let index = jobs.len();
                jobs.push(job);
                outputs.push(T3Output::default());
                started.push(std::time::Instant::now());
                match self.admit(&jobs, index, pair, &mut outputs) {
                    Ok(a) => active[pair] = Some(a),
                    Err(e) => {
                        tracing::warn!(error = %e, "t3: request rejected at admission");
                        on_done(index, std::mem::take(&mut outputs[index]));
                    }
                }
            }
            for pair in 0..active.len() {
                let finished = active[pair].as_ref().is_some_and(|a| a.last == self.c.stop_speech || a.out.len() >= a.max);
                if finished {
                    let a = active[pair].take().expect("checked");
                    let mut o = std::mem::take(&mut outputs[a.job_index]);
                    o.tokens = a.out;
                    o.steps = o.tokens.len();
                    o.decode_us = started[a.job_index].elapsed().as_micros() as u64;
                    on_done(a.job_index, o);
                }
            }
            if active.iter().all(Option::is_none) {
                if closed {
                    return Ok(());
                }
                continue;
            }
            feeds.clear();
            for a in active.iter_mut().flatten() {
                a.out.push(a.last);
                a.history.push(a.last);
                feeds.push((a.cond, a.last));
                feeds.push((a.uncond, a.last));
            }
            self.e.step_slots(&feeds, &mut toks)?;
            for pair in 0..active.len() {
                let Some(a) = active[pair].as_ref() else { continue };
                if a.out.len() >= a.max {
                    continue;
                }
                let (cond, uncond) = (a.cond, a.uncond);
                self.read_logits_row(uncond)?;
                std::mem::swap(&mut self.logits, &mut self.uncond);
                self.read_logits_row(cond)?;
                let a = active[pair].as_mut().expect("checked");
                let u = (!a.greedy).then(|| a.rng.unit());
                a.last = sample_cfg(&self.c, &self.logits, &self.uncond, &a.history, u, &mut self.scratch);
            }
        }
    }
}

'''
s = s[:start] + new + s[end:]
open(p, 'w').write(s)
print("ok")
