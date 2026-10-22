use super::execution_runtime::{ExecutionCommand, ExecutionRequest};
use crate::egress::{EgressConnector, JAVA_TRUSTSTORE};
use crate::guest::GuestState;
use mirror_sandbox::{egress_update, EgressEvent, EgressUpdate};
use mirror_sandbox::{run_command_request, RunCommandRequest, RunCommandResponse};
use mirror_sandbox::{
    sandboxd_server::Sandboxd, CancelRequest, CancelResponse, ExecuteEvent, ExecuteRequest,
};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status};

#[derive(Clone)]
pub(crate) struct SandboxdService {
    pub(crate) commands: mpsc::Sender<ExecutionCommand>,
    pub(crate) guest: Arc<tokio::sync::watch::Sender<GuestState>>,
    pub(crate) closing: tokio::sync::watch::Receiver<bool>,
}
#[tonic::async_trait]
impl Sandboxd for SandboxdService {
    type EgressStream = ReceiverStream<Result<EgressEvent, Status>>;

    async fn egress(
        &self,
        request: Request<tonic::Streaming<EgressUpdate>>,
    ) -> Result<Response<Self::EgressStream>, Status> {
        let mut updates = request.into_inner();
        let update = tokio::time::timeout(std::time::Duration::from_secs(5), updates.message())
            .await
            .map_err(|_| Status::deadline_exceeded("egress configuration timed out"))??;
        let Some(egress_update::Update::Configure(mut config)) =
            update.and_then(|update| update.update)
        else {
            return Err(Status::invalid_argument(
                "egress requires initial configuration",
            ));
        };
        let claimed = self.guest.send_if_modified(|guest| {
            let unconfigured = matches!(guest, GuestState::Unconfigured);
            if unconfigured {
                *guest = GuestState::Preparing;
            }
            unconfigured
        });
        if !claimed {
            return Err(Status::already_exists("egress is already configured"));
        }
        let profiles = std::mem::take(&mut config.profiles);
        let connector =
            match EgressConnector::start(config, std::path::Path::new(JAVA_TRUSTSTORE)).await {
                Ok(connector) => connector,
                Err(error) => {
                    let reason = format!("egress transport could not start: {error}");
                    self.guest.send_replace(GuestState::Failed(reason.clone()));
                    return Err(Status::failed_precondition(reason));
                }
            };
        let (events, receiver) = mpsc::channel(2);
        tokio::spawn(super::egress_setup::run(
            updates,
            events,
            Arc::new(connector),
            profiles,
            self.guest.clone(),
            self.closing.clone(),
        ));
        Ok(Response::new(ReceiverStream::new(receiver)))
    }

    type ExecuteStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<ExecuteEvent, Status>> + Send>>;
    async fn execute(
        &self,
        request: Request<ExecuteRequest>,
    ) -> Result<Response<Self::ExecuteStream>, Status> {
        let (events, receiver) = mpsc::channel(2);
        self.commands
            .send(ExecutionCommand::Execute(ExecutionRequest::Tool {
                request: request.into_inner(),
                events,
            }))
            .await
            .map_err(|_| Status::unavailable("sandboxd execution stopped"))?;
        Ok(Response::new(Box::pin(
            ReceiverStream::new(receiver).map(Ok),
        )))
    }
    type RunCommandStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<RunCommandResponse, Status>> + Send>>;

    async fn run_command(
        &self,
        request: Request<tonic::Streaming<RunCommandRequest>>,
    ) -> Result<Response<Self::RunCommandStream>, Status> {
        let mut input = request.into_inner();
        let first = tokio::time::timeout(std::time::Duration::from_secs(5), input.message())
            .await
            .map_err(|_| Status::deadline_exceeded("command start timed out"))??;
        let Some(run_command_request::Input::Start(request)) =
            first.and_then(|request| request.input)
        else {
            return Err(Status::invalid_argument("expected command start"));
        };
        let (events, receiver) = mpsc::channel(4);
        self.commands
            .send(ExecutionCommand::Execute(ExecutionRequest::Command {
                request,
                input: Box::new(input),
                events,
            }))
            .await
            .map_err(|_| Status::unavailable("sandboxd execution stopped"))?;
        Ok(Response::new(Box::pin(
            ReceiverStream::new(receiver).map(Ok),
        )))
    }

    async fn cancel(
        &self,
        request: Request<CancelRequest>,
    ) -> Result<Response<CancelResponse>, Status> {
        self.commands
            .send(ExecutionCommand::Cancel(request.into_inner().call_id))
            .await
            .map_err(|_| Status::unavailable("sandboxd execution stopped"))?;
        Ok(Response::new(CancelResponse {}))
    }
}
