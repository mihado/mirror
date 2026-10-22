use std::path::PathBuf;
#[derive(Clone, Debug)]
pub struct RunnerCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
}
