use std::sync::Arc;

use mirror_sandbox::{guest_ready, EgressEvent, EgressUpdate, GuestProfile, GuestReady};
use tokio::sync::{mpsc, watch};
use tonic::{Status, Streaming};

use crate::egress::{renew, EgressConnector};
use crate::guest::{Guest, GuestProfiles, GuestState};

pub(crate) async fn run(
    updates: Streaming<EgressUpdate>,
    events: mpsc::Sender<Result<EgressEvent, Status>>,
    connector: Arc<EgressConnector>,
    profiles: Vec<GuestProfile>,
    guest: Arc<watch::Sender<GuestState>>,
    mut closing: watch::Receiver<bool>,
) {
    let preparing = prepare(profiles, connector.clone(), closing.clone());
    let prepared = tokio::select! {
        _ = closing.wait_for(|closed| *closed) => Err("sandboxd closed while applying guest profiles".to_string()),
        prepared = preparing => prepared,
    };
    let outcome = match prepared {
        Ok(prepared) => {
            guest.send_replace(GuestState::Ready(Arc::new(prepared)));
            guest_ready::Outcome::Ready(true)
        }
        Err(reason) => {
            guest.send_replace(GuestState::Failed(reason.clone()));
            guest_ready::Outcome::Failed(reason)
        }
    };
    let ready = matches!(outcome, guest_ready::Outcome::Ready(_));
    let report = EgressEvent {
        ready: Some(GuestReady {
            outcome: Some(outcome),
        }),
        ..EgressEvent::default()
    };
    if events.send(Ok(report)).await.is_ok() && ready {
        renew(updates, events, &connector, closing).await;
    }
    connector.revoke();
}

async fn prepare(
    profiles: Vec<GuestProfile>,
    connector: Arc<EgressConnector>,
    closing: watch::Receiver<bool>,
) -> Result<Guest, String> {
    GuestProfiles::validate(profiles)?
        .ready(connector, closing)
        .await
}
