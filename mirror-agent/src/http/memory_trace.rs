//! Capture committed native records for headless HTTP snapshots.

use axum::body::Bytes;
use futures_util::future::BoxFuture;
use mirror_agent::{AgentMessage, Store, StoreError};
use tokio::sync::watch;

use super::native_jsonl;

/// Add an in-memory JSONL trace to a fresh headless store.
pub struct MemoryTrace {
    store: Box<dyn Store>,
    chunks: watch::Sender<Vec<Bytes>>,
}

impl MemoryTrace {
    /// Return the store wrapper and its read-only trace handle.
    pub fn new(store: Box<dyn Store>) -> (Self, watch::Receiver<Vec<Bytes>>) {
        let (chunks, reader) = watch::channel(Vec::new());
        (Self { store, chunks }, reader)
    }
}

impl Store for MemoryTrace {
    fn load<'a>(
        &'a mut self,
        session_id: &'a str,
        full_history: bool,
    ) -> BoxFuture<'a, Result<Vec<AgentMessage>, StoreError>> {
        self.store.load(session_id, full_history)
    }

    fn append<'a>(
        &'a mut self,
        session_id: &'a str,
        head: u64,
        entries: &'a mut [AgentMessage],
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        Box::pin(async move {
            self.store.append(session_id, head, entries).await?;
            let bytes = native_jsonl(entries).map_err(|error| {
                StoreError::Persistence(format!("could not encode trace: {error}"))
            })?;
            if !bytes.is_empty() {
                self.chunks.send_modify(|chunks| chunks.push(bytes.into()));
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use mirror_agent::react::InMemoryStore;
    use pretty_assertions::assert_eq;

    use super::*;

    #[tokio::test]
    async fn snapshots_preserve_commits_and_exclude_failed_appends() {
        let (mut store, reader) = MemoryTrace::new(Box::new(InMemoryStore::default()));
        let mut first = [AgentMessage::default()];
        store
            .append("session", 0, &mut first)
            .await
            .expect("commit");
        let snapshot = reader.borrow().clone();
        assert_eq!(snapshot.concat(), native_jsonl(&first).expect("encode"));

        let mut second = [AgentMessage::default()];
        store
            .append("session", 1, &mut second)
            .await
            .expect("commit");
        assert_matches!(
            store
                .append("session", 0, &mut [AgentMessage::default()])
                .await,
            Err(StoreError::Stale)
        );
        assert_eq!(snapshot.len(), 1);
        assert_eq!(
            reader.borrow().concat(),
            native_jsonl(&[first, second].concat()).expect("encode"),
        );
    }
}
