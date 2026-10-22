use std::collections::HashMap;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use tokio::sync::watch;

use super::guest_file::write_file;
use super::static_endpoint::{StaticEndpointPlan, StaticEndpointServer};
use crate::egress::{sets_variable, EgressConnector};
use crate::tool_runner::REQUEST_BYTES_VARIABLE;
use mirror_sandbox::{
    environment_variable_name, GuestFile, GuestProfile, SANDBOXD_REFRESH_FILE_CACHE_VARIABLE,
    SANDBOXD_WORKSPACE_VARIABLE,
};

pub(crate) struct GuestProfiles {
    environment: HashMap<String, String>,
    files: Vec<GuestFile>,
    endpoints: Vec<StaticEndpointPlan>,
}

pub(crate) struct Guest {
    egress: Arc<EgressConnector>,
    environment: HashMap<String, String>,
    _endpoints: Vec<StaticEndpointServer>,
}

pub(crate) enum GuestState {
    Unconfigured,
    Preparing,
    Ready(Arc<Guest>),
    Failed(String),
}

const MAX_ENDPOINTS: usize = 8;
const MAX_FILE_BYTES: usize = 4 * 1024 * 1024;
impl GuestProfiles {
    pub(crate) fn validate(profiles: Vec<GuestProfile>) -> Result<Self, String> {
        let mut plan = Self {
            environment: HashMap::new(),
            files: Vec::new(),
            endpoints: Vec::new(),
        };
        for profile in profiles {
            let name = profile.name.clone();
            plan.add(profile)
                .map_err(|problem| format!("guest profile {name:?}: {problem}"))?;
        }
        Ok(plan)
    }

    pub(crate) async fn ready(
        self,
        egress: Arc<EgressConnector>,
        closing: watch::Receiver<bool>,
    ) -> Result<Guest, String> {
        for file in &self.files {
            write_file(file).map_err(|error| format!("could not write {}: {error}", file.path))?;
        }
        let mut endpoints = Vec::new();
        for endpoint in &self.endpoints {
            endpoints.push(endpoint.start(closing.clone()).await?);
        }
        Ok(Guest {
            egress,
            environment: self.environment,
            _endpoints: endpoints,
        })
    }

    fn add(&mut self, profile: GuestProfile) -> Result<(), String> {
        for (name, value) in profile.environment {
            self.add_variable(name, value)?;
        }
        for file in profile.files {
            self.add_file(file)?;
        }
        for endpoint in profile.endpoints {
            if self.endpoints.len() >= MAX_ENDPOINTS {
                return Err(format!("at most {MAX_ENDPOINTS} endpoints are allowed"));
            }
            self.endpoints.push(StaticEndpointPlan::new(endpoint)?);
        }
        Ok(())
    }

    fn add_variable(&mut self, name: String, value: String) -> Result<(), String> {
        if !environment_variable_name(&name) || value.contains('\0') {
            return Err(format!(
                "variable {name:?} is not a valid environment variable"
            ));
        }
        if sets_variable(&name)
            || [
                REQUEST_BYTES_VARIABLE,
                SANDBOXD_WORKSPACE_VARIABLE,
                SANDBOXD_REFRESH_FILE_CACHE_VARIABLE,
            ]
            .contains(&name.as_str())
        {
            return Err(format!("variable {name:?} is owned by sandboxd"));
        }
        if self.environment.contains_key(&name) {
            return Err(format!("variable {name:?} is set by more than one profile"));
        }
        self.environment.insert(name, value);
        Ok(())
    }

    fn add_file(&mut self, file: GuestFile) -> Result<(), String> {
        let path = Path::new(&file.path);
        if !path.is_absolute() || !(1..=0o777).contains(&file.mode) {
            return Err(format!(
                "file {:?} needs an absolute path and a mode",
                file.path
            ));
        }
        if self
            .files
            .iter()
            .any(|other| Path::new(&other.path) == path)
        {
            return Err(format!("file {:?} is written twice", file.path));
        }
        let total = self
            .files
            .iter()
            .map(|other| other.contents.len())
            .sum::<usize>();
        if total + file.contents.len() > MAX_FILE_BYTES {
            return Err(format!("files exceed {MAX_FILE_BYTES} bytes"));
        }
        self.files.push(file);
        Ok(())
    }
}

impl Guest {
    pub(crate) fn configure_command(&self, command: &mut Command) {
        self.egress.configure_command(command);
        command.envs(&self.environment);
    }
}

impl GuestState {
    pub(crate) fn admit(&self) -> Result<Option<Arc<Guest>>, String> {
        match self {
            Self::Unconfigured => Ok(None),
            Self::Ready(guest) => Ok(Some(guest.clone())),
            Self::Preparing => Err("sandbox guest profiles are still being applied".into()),
            Self::Failed(reason) => Err(format!("sandbox guest is not ready: {reason}")),
        }
    }
}

#[cfg(test)]
#[path = "guest_profiles_tests.rs"]
mod tests;
