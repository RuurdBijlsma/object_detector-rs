use crate::error::ObjectDetectorError;
use ort::environment::Environment;
use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

#[derive(Debug)]
pub struct OnnxSession {
    pub session: Session,
}

impl OnnxSession {
    pub fn new(
        path: impl AsRef<Path>,
        execution_providers: &[ExecutionProviderDispatch],
        optimization_level: Option<GraphOptimizationLevel>,
        intra_threads: Option<usize>,
        inter_threads: Option<usize>,
        memory_pattern: Option<bool>,
        environment: Option<&Environment>,
    ) -> Result<Self, ObjectDetectorError> {
        let environment = get_environment(environment)?;
        let mut session_builder = Session::builder(&environment)?;
        if !execution_providers.is_empty() {
            session_builder = session_builder.with_execution_providers(execution_providers)?;
        }
        if let Some(opt_level) = optimization_level {
            session_builder = session_builder.with_optimization_level(opt_level)?;
        }
        if let Some(intra) = intra_threads {
            session_builder = session_builder.with_intra_threads(intra)?;
        }
        if let Some(inter) = inter_threads {
            session_builder = session_builder.with_inter_threads(inter)?;
        }
        if let Some(mem_pattern) = memory_pattern {
            session_builder = session_builder.with_memory_pattern(mem_pattern)?;
        }
        let session = session_builder.commit_from_file(path)?;

        Ok(Self { session })
    }
}

pub(crate) fn get_environment(
    environment: Option<&Environment>,
) -> Result<Environment, ObjectDetectorError> {
    environment.map_or_else(default_environment, |environment| Ok(environment.clone()))
}

/// Reuse the process-wide default environment for sessions without a caller-supplied one.
fn default_environment() -> Result<Environment, ObjectDetectorError> {
    static ENVIRONMENT: Mutex<Option<Environment>> = Mutex::new(None);
    let mut slot = ENVIRONMENT.lock().unwrap_or_else(PoisonError::into_inner);
    let environment = if let Some(environment) = slot.as_ref() {
        environment.clone()
    } else {
        let environment = ort::init().with_name("object_detector").build()?;
        *slot = Some(environment.clone());
        environment
    };
    drop(slot);
    Ok(environment)
}
