//! Append-only native conversation traces for a single server run.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use futures_util::future::BoxFuture;
use mirror_agent::react::InMemoryStore;
use mirror_agent::AgentMessage;
use mirror_agent::{Store, StoreError};

use super::native_jsonl;

/// Keep live history in memory and flush each committed record to JSONL.
pub struct JsonlStore {
    memory: InMemoryStore,
    file: Option<File>,
}

impl JsonlStore {
    /// Create a fresh journal before starting the agent.
    pub fn create(output: &Path) -> std::io::Result<Self> {
        Ok(Self {
            memory: InMemoryStore::default(),
            file: Some(File::create(output)?),
        })
    }

    fn write(&mut self, entries: &[AgentMessage]) -> Result<(), StoreError> {
        // A failed append may leave a partial line; never append after it.
        let mut file = self
            .file
            .take()
            .ok_or_else(|| persistence("trace journal is closed after a write failure"))?;
        let buffer = native_jsonl(entries).map_err(persistence)?;
        file.write_all(&buffer).map_err(persistence)?;
        file.flush().map_err(persistence)?;
        self.file = Some(file);
        Ok(())
    }
}

impl Store for JsonlStore {
    fn load<'a>(
        &'a mut self,
        session_id: &'a str,
        full_history: bool,
    ) -> BoxFuture<'a, Result<Vec<AgentMessage>, StoreError>> {
        self.memory.load(session_id, full_history)
    }

    fn append<'a>(
        &'a mut self,
        session_id: &'a str,
        head: u64,
        entries: &'a mut [AgentMessage],
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        Box::pin(async move {
            // InMemoryStore is immediately ready; don't yield between memory and journal writes.
            self.memory.append(session_id, head, entries).await?;
            self.write(entries)
        })
    }
}

fn persistence(error: impl std::fmt::Display) -> StoreError {
    StoreError::Persistence(format!("could not write trace: {error}"))
}
