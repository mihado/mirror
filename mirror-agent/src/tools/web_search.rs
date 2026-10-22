//! Provider-independent web search formatting.

use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;

use super::object_schema;
use super::Tool;
use super::ToolError;
use crate::tools::ToolOutput;
use crate::web::{SearchResult, WebProviderFactory};
use crate::ExecutionContext;

/// Search public webpages using a configured retrieval provider.
pub struct WebSearchTool {
    make_provider: Box<WebProviderFactory>,
    name: &'static str,
}

/// Arguments accepted by web search.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WebSearchArgs {
    pub query: String,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<ModelSearchResult>,
}

// Field order is part of the model-visible JSON contract.
#[derive(Serialize)]
struct ModelSearchResult {
    title: Option<String>,
    link: String,
    snippet: Option<String>,
}

const NUM_RESULTS: usize = 10;
const MAX_OUTPUT_CHARS: usize = 20_000;
const MAX_TITLE_CHARS: usize = 500;
const MAX_SNIPPET_CHARS: usize = 1_500;
const MAX_URL_CHARS: usize = 2_048;

impl WebSearchTool {
    /// Construct the selected adapter for each search request.
    pub fn new(make_provider: Box<WebProviderFactory>) -> Self {
        Self {
            make_provider,
            name: "web_search",
        }
    }

    /// Override the model-facing tool name.
    #[must_use]
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }
}

impl Tool for WebSearchTool {
    type Args = WebSearchArgs;

    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "Search for content on the internet. Results include the website title, address (URL), and description."
    }

    fn parameters(&self) -> Map<String, Value> {
        object_schema(
            Map::from_iter([(
                "query".to_string(),
                serde_json::json!({"type": "string", "description": "The search query."}),
            )]),
            &["query"],
        )
    }

    fn supports_remote(&self) -> bool {
        true
    }

    fn is_concurrency_safe(&self) -> bool {
        true
    }

    async fn run_remote(
        &self,
        args: Self::Args,
        context: &ExecutionContext,
        _remote: &super::RemoteToolExecutor,
    ) -> Result<ToolOutput, ToolError> {
        // This tool uses a narrow host service, not the workspace or remote transport.
        self.run(args, context).await
    }

    async fn run(
        &self,
        args: WebSearchArgs,
        context: &ExecutionContext,
    ) -> Result<ToolOutput, ToolError> {
        let client = match (self.make_provider)(context.web_provider()) {
            Ok(provider) => provider,
            Err(error) => return Ok(ToolOutput::error(error.to_string())),
        };
        let results = match client.search(&args.query, NUM_RESULTS).await {
            Ok(results) => results,
            Err(error) => return Ok(ToolOutput::error(error.to_string())),
        };
        let (content, count) = serialize_results(results)?;
        Ok(ToolOutput::success(content)
            .with_display(format!("{}: {count} result(s)", client.name())))
    }
}

fn serialize_results(results: Vec<SearchResult>) -> Result<(String, usize), ToolError> {
    let normalized = results
        .into_iter()
        .filter_map(|result| {
            let link = result_link(&result.link)?;
            Some(ModelSearchResult {
                title: bounded_text(result.title.as_deref(), MAX_TITLE_CHARS),
                link,
                snippet: bounded_text(result.snippet.as_deref(), MAX_SNIPPET_CHARS),
            })
        })
        .collect::<Vec<_>>();
    let mut response = SearchResponse {
        results: normalized,
    };
    loop {
        let content = serde_json::to_string_pretty(&response)
            .map_err(|error| ToolError::Execution(error.to_string()))?;
        if content.chars().count() <= MAX_OUTPUT_CHARS || response.results.is_empty() {
            return Ok((content, response.results.len()));
        }
        response.results.pop();
    }
}

fn result_link(link: &str) -> Option<String> {
    let link = link.trim();
    if link.is_empty() || link.chars().count() > MAX_URL_CHARS {
        return None;
    }
    let parsed = reqwest::Url::parse(link).ok()?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return None;
    }
    Some(link.to_string())
}

fn bounded_text(text: Option<&str>, limit: usize) -> Option<String> {
    let text = text?;
    if text.chars().count() <= limit {
        return Some(text.to_string());
    }
    let mut bounded = text.chars().take(limit - 1).collect::<String>();
    bounded = bounded.trim_end().to_string();
    bounded.push('…');
    Some(bounded)
}

#[cfg(test)]
#[path = "web_search_test.rs"]
mod tests;
