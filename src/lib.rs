#![allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

mod error;
pub mod model_manager;
pub mod object_detector;
pub mod predictor;
mod structs;

pub use error::ObjectDetectorError;
pub use object_detector::ObjectDetector;
pub use predictor::{PromptFreeDetector, PromptableDetector, YoloPreprocessMeta};
pub use structs::*;
