//! Linear model conversations, separate from the native state journal.

use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::body::Bytes;
use futures_util::{Stream, StreamExt};
use mirror_agent::chat_record;
use mirror_agent::providers::{Completion, CompletionRequest};
use mirror_agent::providers::{Llm, LlmError, StreamEvent};
use prost::{Message, Name};
use prost_reflect::SerializeOptions;
use serde::ser::{SerializeMap, Serializer};
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio::sync::{mpsc, oneshot, watch};

use super::dynamic_message;

/// Record model inputs and completed outputs without changing inference behavior.
pub struct TracedModel<Model> {
    model: Model,
    records: mpsc::Sender<(CallEvent, oneshot::Sender<()>)>,
}

#[derive(Debug)]
enum CallEvent {
    Request(CompletionRequest),
    Response(Completion),
    Error(String),
}

impl<Model: Llm> TracedModel<Model> {
    /// Start one diagnostic writer for the requested file and memory outputs.
    pub async fn create(
        model: Model,
        path: Option<&Path>,
        memory: Option<watch::Sender<Vec<Bytes>>>,
    ) -> io::Result<Self> {
        let mut file = match path {
            Some(path) => Some(
                tokio::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .mode(0o600)
                    .open(path)
                    .await?,
            ),
            None => None,
        };
        let (records, mut receiver) = mpsc::channel::<(CallEvent, oneshot::Sender<()>)>(1);
        let model_name = model.model().to_string();
        tokio::spawn(async move {
            let mut context = None;
            while let Some((event, written)) = receiver.recv().await {
                if let Err(error) = async {
                    let records = call_records(event, &mut context, &model_name)?;
                    let mut buffer = Vec::new();
                    for record in records {
                        write_record(&mut buffer, record)?;
                    }
                    if let Some(output) = &mut file {
                        if let Err(error) = async {
                            output.write_all(&buffer).await?;
                            output.flush().await
                        }
                        .await
                        {
                            eprintln!(
                                "model-call file trace disabled after write failure: {error}"
                            );
                            file = None;
                        }
                    }
                    if let Some(memory) = &memory {
                        if !buffer.is_empty() {
                            memory.send_modify(|chunks| chunks.push(buffer.into()));
                        }
                    }
                    Ok::<_, io::Error>(())
                }
                .await
                {
                    // Never append after a possibly partial line or interrupt inference for logging.
                    eprintln!("model-call trace disabled after write failure: {error}");
                    break;
                }
                let _ = written.send(());
                if file.is_none() && memory.is_none() {
                    break;
                }
            }
        });
        Ok(Self { model, records })
    }

    async fn record(&self, event: CallEvent) {
        let (written, flushed) = oneshot::channel();
        if self.records.send((event, written)).await.is_ok() {
            let _ = flushed.await;
        }
    }
}

impl<Model: Llm> Llm for TracedModel<Model> {
    fn model(&self) -> &str {
        self.model.model()
    }

    fn prewarm(&self, request: CompletionRequest) {
        self.model.prewarm(request);
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Stream<Item = Result<StreamEvent, LlmError>> + Send {
        stream! {
            self.record(CallEvent::Request(request.clone())).await;
            let events = self.model.stream(request);
            futures_util::pin_mut!(events);
            let mut finished = false;
            while let Some(event) = events.next().await {
                match &event {
                    Ok(StreamEvent::Final(completion)) => {
                        self.record(CallEvent::Response(completion.clone())).await;
                        finished = true;
                    }
                    Err(error) => {
                        self.record(CallEvent::Error(error.to_string())).await;
                        finished = true;
                    }
                    _ => {}
                }
                yield event;
            }
            if !finished {
                self.record(CallEvent::Error(
                    "model stream ended without a final response".to_string(),
                )).await;
            }
        }
    }
}

fn call_records(
    event: CallEvent,
    context: &mut Option<CompletionRequest>,
    model: &str,
) -> io::Result<Vec<Value>> {
    match event {
        CallEvent::Request(request) => {
            let records = request_records(&request, context.as_ref(), model)?;
            *context = Some(request);
            Ok(records)
        }
        CallEvent::Response(completion) => {
            let message = chat_record::Message::Assistant(completion.message);
            let mut record = message_json(&message)?;
            record["usage"] = completion
                .usage
                .as_ref()
                .map(proto_json)
                .transpose()?
                .into();
            record["truncated"] = completion.truncated.into();
            if let Some(context) = context {
                context.messages.push(message);
            }
            Ok(vec![record])
        }
        CallEvent::Error(message) => Ok(vec![json!({"event": "error", "message": message})]),
    }
}

fn request_records(
    request: &CompletionRequest,
    previous: Option<&CompletionRequest>,
    model: &str,
) -> io::Result<Vec<Value>> {
    let continuing = previous.filter(|previous| {
        request.messages.starts_with(&previous.messages)
            && request.tools == previous.tools
            && request.max_tokens == previous.max_tokens
            && request.inference == previous.inference
    });
    let start = continuing.map_or(0, |previous| previous.messages.len());
    let mut records = request.messages[start..]
        .iter()
        .map(message_json)
        .collect::<io::Result<Vec<_>>>()?;
    if continuing.is_none() {
        if let Some(system) = records.first_mut() {
            system["model"] = model.into();
            system["tools"] = json!(request.tools);
            system["max_tokens"] = request.max_tokens.into();
            system["inference"] = request
                .inference
                .as_ref()
                .map(proto_json)
                .transpose()?
                .into();
        }
        if previous.is_some() {
            records.insert(0, json!({"event": "context_reset"}));
        }
    }
    Ok(records)
}

fn message_json(message: &chat_record::Message) -> io::Result<Value> {
    let (role, mut record) = match message {
        chat_record::Message::System(message) => ("system", proto_json(message)?),
        chat_record::Message::User(message) => ("user", proto_json(message)?),
        chat_record::Message::Assistant(message) => ("assistant", proto_json(message)?),
        chat_record::Message::Tool(message) => ("tool", proto_json(message)?),
    };
    record["role"] = role.into();
    Ok(record)
}

fn proto_json<M: Message + Name>(message: &M) -> io::Result<Value> {
    dynamic_message(message)?
        .serialize_with_options(
            serde_json::value::Serializer,
            &SerializeOptions::new()
                .use_proto_field_name(true)
                .stringify_64_bit_integers(false),
        )
        .map_err(io::Error::other)
}

fn write_record(buffer: &mut Vec<u8>, record: Value) -> io::Result<()> {
    let Value::Object(mut fields) = record else {
        return Err(io::Error::other("model-call record must be an object"));
    };
    let mut serializer = serde_json::Serializer::new(&mut *buffer);
    let mut map = serializer.serialize_map(None)?;
    for key in [
        "role",
        "event",
        "content",
        "reasoning_content",
        "reasoning_details",
        "tool_calls",
        "tool_call_id",
        "name",
        "message",
        "tools",
        "model",
        "max_tokens",
        "inference",
        "usage",
        "truncated",
    ] {
        if let Some(value) = fields.remove(key) {
            map.serialize_entry(key, &value)?;
        }
    }
    for (key, value) in fields {
        map.serialize_entry(&key, &value)?;
    }
    map.serialize_entry(
        "timestamp_ms",
        &SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    )?;
    map.end()?;
    buffer.push(b'\n');
    Ok(())
}
