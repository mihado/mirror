//! Typed tool execution primitives.

use std::future::Future;

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;
use thiserror::Error;

use crate::tools::ToolOutput;
use crate::ExecutionContext;

/// A tool with typed JSON arguments and asynchronous execution.
pub trait Tool: Send + Sync {
    type Args: DeserializeOwned + Serialize + Send;

    /// The model-facing tool name.
    fn name(&self) -> &str;

    /// A natural-language description of what the tool does.
    fn description(&self) -> &str;

    /// The JSON Schema used to validate model-generated arguments.
    fn parameters(&self) -> Map<String, Value>;

    /// Whether calls require human approval.
    fn needs_approval(&self) -> bool {
        false
    }

    fn always_asks(&self) -> bool {
        false
    }

    /// Whether this tool and its descendants support remote execution.
    fn supports_remote(&self) -> bool {
        false
    }

    /// Whether calls may run at the same time as other concurrency-safe calls.
    ///
    /// Return true only when a call changes nothing that another call could
    /// observe. Other calls run alone, after every earlier call in the batch.
    fn is_concurrency_safe(&self) -> bool {
        false
    }

    /// Execute remotely; unsupported tools fail without touching the host workspace.
    fn run_remote(
        &self,
        _args: Self::Args,
        _context: &ExecutionContext,
        _remote: &super::RemoteToolExecutor,
    ) -> impl Future<Output = Result<ToolOutput, ToolError>> + Send {
        async {
            Err(ToolError::Execution(format!(
                "tool {:?} has no remote implementation",
                self.name()
            )))
        }
    }

    /// Execute one validated call.
    fn run(
        &self,
        args: Self::Args,
        context: &ExecutionContext,
    ) -> impl Future<Output = Result<ToolOutput, ToolError>> + Send;
}

/// A typed failure produced while validating or running a tool.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolError {
    #[error("no connection for {0}")]
    NotConnected(String),
    #[error("invalid tool arguments: {0}")]
    InvalidArguments(String),
    #[error("tool execution failed: {0}")]
    Execution(String),
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::{assert_eq, assert_ne};
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct EchoArgs {
        text: String,
    }

    struct EchoTool;

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct EmptyArgs {}

    struct EmptyTool;

    impl Tool for EchoTool {
        type Args = EchoArgs;

        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echo text."
        }

        fn parameters(&self) -> Map<String, Value> {
            Map::from_iter([
                ("type".to_string(), Value::String("object".to_string())),
                (
                    "properties".to_string(),
                    serde_json::json!({"text": {"type": "string"}}),
                ),
                ("required".to_string(), serde_json::json!(["text"])),
                ("additionalProperties".to_string(), Value::Bool(false)),
            ])
        }

        async fn run(
            &self,
            args: Self::Args,
            context: &ExecutionContext,
        ) -> Result<ToolOutput, ToolError> {
            assert!(!context.tool_permissions().allow_outside_workspace());
            Ok(ToolOutput::success(args.text))
        }
    }

    impl Tool for EmptyTool {
        type Args = EmptyArgs;

        fn name(&self) -> &str {
            "empty"
        }

        fn description(&self) -> &str {
            "Accept no arguments."
        }

        fn parameters(&self) -> Map<String, Value> {
            Map::from_iter([
                ("type".to_string(), Value::String("object".to_string())),
                ("properties".to_string(), Value::Object(Map::new())),
                ("additionalProperties".to_string(), Value::Bool(false)),
            ])
        }

        async fn run(
            &self,
            _args: Self::Args,
            _context: &ExecutionContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success("ran"))
        }
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct TypedArgs {
        count: u32,
        ratio: f64,
        flag: bool,
        items: Vec<String>,
        label: String,
        limit: Option<u32>,
    }

    struct TypedTool;

    impl Tool for TypedTool {
        type Args = TypedArgs;

        fn name(&self) -> &str {
            "typed"
        }

        fn description(&self) -> &str {
            "Echo typed arguments."
        }

        fn parameters(&self) -> Map<String, Value> {
            Map::from_iter([
                ("type".to_string(), Value::String("object".to_string())),
                (
                    "properties".to_string(),
                    serde_json::json!({
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "flag": {"type": "boolean"},
                        "items": {"type": "array", "items": {"type": "string"}},
                        "label": {"type": "string"},
                        "limit": {"type": ["integer", "null"]},
                    }),
                ),
            ])
        }

        async fn run(
            &self,
            args: Self::Args,
            _context: &ExecutionContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success(
                serde_json::to_string(&args).expect("serialize arguments"),
            ))
        }
    }

    #[tokio::test]
    async fn execute_coerces_stringified_arguments_to_their_schema_types() {
        let result = crate::tools::AnyTool::execute(
            &TypedTool,
            r#"{"count":"15.0","ratio":"0.5","flag":"TRUE","items":["a"],"label":"7","limit":10.0}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_eq!(
            result,
            ToolOutput::success(
                r#"{"count":15,"ratio":0.5,"flag":true,"items":["a"],"label":"7","limit":10}"#
            )
        );
    }

    #[tokio::test]
    async fn execute_rejects_strings_that_are_not_a_value_of_the_schema_type() {
        let result = crate::tools::AnyTool::execute(
            &TypedTool,
            r#"{"count":"1.5","ratio":1,"flag":true,"items":[],"label":"x","limit":null}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_matches!(result, ToolOutput { success: false, .. });
        assert!(result
            .content
            .starts_with("Invalid arguments for tool typed: invalid type: string \"1.5\""));
    }

    #[test]
    fn definition_exposes_function_schema() {
        let definition = crate::tools::AnyTool::definition(&EchoTool);

        assert_eq!(definition.name, "echo");
        assert_eq!(definition.description, "Echo text.");
        assert_eq!(definition.parameters["additionalProperties"], false);
    }

    #[test]
    fn executor_preserves_tool_definitions() {
        use crate::tools::{AnyTool, DirectToolExecutor, ReadTool, ToolExecutor};

        let workspace = tempfile::tempdir().expect("workspace");
        let read = ReadTool::new(workspace.path()).expect("read tool");
        let raw = AnyTool::definition(&read);
        let executor = DirectToolExecutor::new(vec![
            std::sync::Arc::new(read),
            std::sync::Arc::new(EchoTool),
        ]);

        let definitions = executor.definitions();
        assert_eq!(definitions[0], raw);
        assert_eq!(
            definitions[0].description,
            "Read a UTF-8 text file from the workspace."
        );
        assert_eq!(definitions[1], AnyTool::definition(&EchoTool));
        assert_eq!(executor.definitions(), definitions);
    }

    #[tokio::test]
    async fn execute_validates_and_runs_typed_arguments() {
        let result = crate::tools::AnyTool::execute(
            &EchoTool,
            r#"{"text":"hello"}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_ne!(result, ToolOutput::error("hello"));
        assert_eq!(result, ToolOutput::success("hello"));
    }

    #[tokio::test]
    async fn execute_treats_empty_arguments_as_an_empty_object() {
        let result =
            crate::tools::AnyTool::execute(&EmptyTool, "  ", &ExecutionContext::default()).await;

        assert_ne!(result, ToolOutput::error("ran"));
        assert_eq!(result, ToolOutput::success("ran"));
        assert_matches!(result, ToolOutput { success: true, .. });
    }

    #[tokio::test]
    async fn execute_returns_invalid_json_as_a_tool_error() {
        let result = crate::tools::AnyTool::execute(
            &EchoTool,
            r#"{"unknown":true}"#,
            &ExecutionContext::default(),
        )
        .await;

        assert_matches!(result, ToolOutput { success: false, .. });
        assert!(result
            .content
            .starts_with("Invalid arguments for tool echo: unknown field `unknown`"));
    }
}
