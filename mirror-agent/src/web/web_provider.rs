//! Web retrieval boundary, independent of model providers and tool formatting.

use std::fmt::Debug;

use async_trait::async_trait;
use thiserror::Error;

use super::{FirecrawlError, TavilyError, YouError};

/// A paired search and full-page retrieval backend.
#[async_trait]
pub trait WebProvider: Debug + Send + Sync {
    /// Provider label for human-facing displays, never a tool argument.
    fn name(&self) -> &'static str;

    /// Whether this backend has a non-blank credential.
    fn is_configured(&self) -> bool;

    /// Return results in provider rank order, before model-facing output limits.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, WebError>;

    /// Return full extracted text, not query-ranked snippets or a preview.
    async fn contents(&self, url: &str) -> Result<Page, WebError>;
}

/// Normalized search data shared with the model-facing search tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    pub title: Option<String>,
    pub link: String,
    pub snippet: Option<String>,
}

/// One normalized page consumed by the browsing tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    pub url: String,
    pub title: Option<String>,
    pub text: String,
}

/// Safe provider failures; adapters must not include bodies, keys, or headers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WebError {
    #[error(transparent)]
    You(#[from] YouError),
    #[error(transparent)]
    Tavily(#[from] TavilyError),
    #[error(transparent)]
    Firecrawl(#[from] FirecrawlError),
}
