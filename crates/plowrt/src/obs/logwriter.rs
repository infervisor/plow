//! Non-blocking log sink for `serve`: formatted lines go through a bounded channel to one
//! writer thread, so an engine-thread log line never waits on stdout. A pipe with a slow reader
//! otherwise stalls the serving tick for the duration of the `write(2)`.
//!
//! What is never lost: `warn!`/`error!` bypass the queue and are written synchronously, and
//! [`flush`] drains the queue on every exit path (shutdown, engine death, panic). What may be
//! lost: `info!` and below when the queue is full — each dropped line is counted and reported.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use tracing::{Level, Metadata};

const QUEUE_LINES: usize = 16384;

static INSTALLED: OnceLock<NonBlockingStdout> = OnceLock::new();

enum Msg {
    Line(Vec<u8>),
    /// Write everything queued before this marker, flush stdout, then ack.
    Flush(SyncSender<()>),
}

#[derive(Clone)]
pub struct NonBlockingStdout {
    tx: SyncSender<Msg>,
    dropped: Arc<AtomicU64>,
}

impl NonBlockingStdout {
    /// Spawns the writer thread and registers the sink for [`flush`].
    pub fn install() -> Self {
        let w = Self::spawn();
        let _ = INSTALLED.set(w.clone());
        w
    }

    fn spawn() -> Self {
        let (tx, rx) = sync_channel::<Msg>(QUEUE_LINES);
        let dropped = Arc::new(AtomicU64::new(0));
        let seen = Arc::clone(&dropped);
        let spawned = std::thread::Builder::new()
            .name("plowrt-log".into())
            .spawn(move || {
                let stdout = std::io::stdout();
                let mut reported = 0u64;
                for msg in rx {
                    let mut out = stdout.lock();
                    match msg {
                        Msg::Line(line) => {
                            let _ = out.write_all(&line);
                        }
                        Msg::Flush(ack) => {
                            let _ = out.flush();
                            let _ = ack.send(());
                        }
                    }
                    let n = seen.load(Ordering::Relaxed);
                    if n != reported {
                        let _ = writeln!(
                            out,
                            "plowrt-log: {} line(s) dropped, queue full",
                            n - reported
                        );
                        reported = n;
                    }
                }
            });
        if spawned.is_err() {
            eprintln!("plowrt-log: writer thread failed to start; logging synchronously");
        }
        Self { tx, dropped }
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Waits until every line queued so far is written, or `timeout` passes.
    pub fn flush(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let (ack_tx, ack_rx) = sync_channel(1);
        let mut marker = Msg::Flush(ack_tx);
        loop {
            match self.tx.try_send(marker) {
                Ok(()) => break,
                Err(TrySendError::Disconnected(_)) => return false,
                Err(TrySendError::Full(m)) => {
                    if Instant::now() >= deadline {
                        return false;
                    }
                    marker = m;
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
        ack_rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .is_ok()
    }
}

/// Drains the installed sink; a direct stdout flush when `serve` logging is not installed.
/// Call before every `process::exit` so the last lines of a shutdown or fault are on disk.
pub fn flush(timeout: Duration) -> bool {
    crate::obs::pfx_trace::flush(timeout);
    match INSTALLED.get() {
        Some(w) => w.flush(timeout),
        None => std::io::stdout().flush().is_ok(),
    }
}

pub enum LogWriter {
    Queued(NonBlockingStdout),
    /// `warn!`/`error!`: written in place, never dropped. May land ahead of queued `info!` lines
    /// (the timestamps order them).
    Direct,
}

impl Write for LogWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            LogWriter::Direct => std::io::stdout().lock().write_all(buf)?,
            LogWriter::Queued(q) => match q.tx.try_send(Msg::Line(buf.to_vec())) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    q.dropped.fetch_add(1, Ordering::Relaxed);
                }
                // The writer thread is gone (failed to start): write in place.
                Err(TrySendError::Disconnected(Msg::Line(line))) => {
                    std::io::stdout().lock().write_all(&line)?
                }
                Err(TrySendError::Disconnected(Msg::Flush(_))) => {}
            },
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for NonBlockingStdout {
    type Writer = LogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        LogWriter::Queued(self.clone())
    }

    fn make_writer_for(&'a self, meta: &Metadata<'_>) -> Self::Writer {
        if *meta.level() <= Level::WARN {
            LogWriter::Direct
        } else {
            LogWriter::Queued(self.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::fmt::MakeWriter;

    #[test]
    fn write_never_blocks_and_counts_drops_when_full() {
        let (tx, _rx) = sync_channel::<Msg>(2);
        let q = NonBlockingStdout {
            tx,
            dropped: Arc::new(AtomicU64::new(0)),
        };
        let mut w = LogWriter::Queued(q.clone());
        for _ in 0..5 {
            assert_eq!(w.write(b"line\n").unwrap(), 5);
        }
        assert_eq!(q.dropped(), 3);
    }

    #[test]
    fn flush_waits_for_queued_lines() {
        let q = NonBlockingStdout::spawn();
        let mut w = LogWriter::Queued(q.clone());
        for _ in 0..100 {
            w.write_all(b"queued line\n").unwrap();
        }
        assert!(q.flush(Duration::from_secs(5)));
        assert_eq!(q.dropped(), 0);
    }

    #[test]
    fn warnings_and_errors_bypass_the_queue() {
        let q = NonBlockingStdout::spawn();
        let direct = |level: Level| {
            let cs = tracing::callsite::Identifier(&CS);
            let meta = Metadata::new(
                "t",
                "t",
                level,
                None,
                None,
                None,
                tracing::field::FieldSet::new(&[], cs),
                tracing::metadata::Kind::EVENT,
            );
            matches!(q.make_writer_for(&meta), LogWriter::Direct)
        };
        assert!(direct(Level::ERROR));
        assert!(direct(Level::WARN));
        assert!(!direct(Level::INFO));
        assert!(!direct(Level::DEBUG));
    }

    struct TestCallsite;
    static CS: TestCallsite = TestCallsite;
    impl tracing::Callsite for TestCallsite {
        fn set_interest(&self, _: tracing::subscriber::Interest) {}
        fn metadata(&self) -> &Metadata<'_> {
            unreachable!()
        }
    }
}
