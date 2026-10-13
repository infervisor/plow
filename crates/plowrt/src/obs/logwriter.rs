//! Non-blocking log sink for `serve`: formatted lines go through a bounded channel to one
//! writer thread, so an engine-thread log line never waits on stdout. A pipe with a slow reader
//! otherwise stalls the serving tick for the duration of the `write(2)`. When the queue is full
//! the line is dropped and counted; dropping a log line is cheaper than stalling a tick.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::Arc;

const QUEUE_LINES: usize = 16384;

#[derive(Clone)]
pub struct NonBlockingStdout {
    tx: SyncSender<Vec<u8>>,
    dropped: Arc<AtomicU64>,
}

impl NonBlockingStdout {
    /// Spawns the writer thread. The thread ends with the process; lines still queued at exit
    /// are lost, which is acceptable for a long-running server.
    pub fn spawn() -> Self {
        let (tx, rx) = sync_channel::<Vec<u8>>(QUEUE_LINES);
        let dropped = Arc::new(AtomicU64::new(0));
        let seen = Arc::clone(&dropped);
        let spawned = std::thread::Builder::new()
            .name("plowrt-log".into())
            .spawn(move || {
                let stdout = std::io::stdout();
                let mut reported = 0u64;
                for line in rx {
                    let mut out = stdout.lock();
                    let _ = out.write_all(&line);
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
}

impl Write for NonBlockingStdout {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self.tx.try_send(buf.to_vec()) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
            // The writer thread is gone (failed to start): fall back to a direct write.
            Err(TrySendError::Disconnected(line)) => {
                let _ = std::io::stdout().lock().write_all(&line);
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for NonBlockingStdout {
    type Writer = NonBlockingStdout;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_never_blocks_and_counts_drops_when_full() {
        let (tx, _rx) = sync_channel::<Vec<u8>>(2);
        let mut w = NonBlockingStdout {
            tx,
            dropped: Arc::new(AtomicU64::new(0)),
        };
        for _ in 0..5 {
            assert_eq!(w.write(b"line\n").unwrap(), 5);
        }
        assert_eq!(w.dropped(), 3);
    }
}
