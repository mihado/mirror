//! In-memory retrieval for shared tool and routing tests.

use async_trait::async_trait;

use crate::web::{Page, SearchResult, WebError, WebProvider, WebProviderConfig};

#[derive(Debug)]
pub struct MemoryWebProvider(pub Option<WebProviderConfig>);

#[async_trait]
impl WebProvider for MemoryWebProvider {
    fn name(&self) -> &'static str {
        match self.0.unwrap_or_default() {
            WebProviderConfig::You => "You.com",
            WebProviderConfig::Tavily => "Tavily",
            WebProviderConfig::Firecrawl => "Firecrawl",
        }
    }

    fn is_configured(&self) -> bool {
        true
    }

    async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<SearchResult>, WebError> {
        Ok(vec![SearchResult {
            title: Some(self.name().to_string()),
            link: "https://example.com/final".to_string(),
            snippet: None,
        }])
    }

    async fn contents(&self, _url: &str) -> Result<Page, WebError> {
        Ok(Page {
            url: "https://example.com/final".to_string(),
            title: None,
            text: self.name().to_string(),
        })
    }
}
