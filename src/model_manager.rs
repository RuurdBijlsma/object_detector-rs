use crate::ObjectDetectorError;
use crate::structs::{DetectorType, ModelScale};
#[cfg(feature = "hf-hub")]
use hf_hub::{HFClient, split_id};
use std::path::{Path, PathBuf};

/// Details for fetching model files from `HuggingFace` Hub.
pub struct HfModel {
    /// Repository ID (e.g., "user/repo")
    pub id: String,
    /// Filename within the repository
    pub file: String,
}

impl HfModel {
    pub const DEFAULT_REPO_ID: &'static str = "RuteNL/yolo26-object-detection-ONNX";
    pub const DEFAULT_CLIP_REPO: &'static str = "RuteNL/MobileCLIP2-B-OpenCLIP-ONNX";

    #[must_use]
    pub fn get_model_file_path(
        detector_type: DetectorType,
        scale: ModelScale,
        include_mask: bool,
    ) -> String {
        let folder = match detector_type {
            #[cfg(feature = "promptable")]
            DetectorType::Promptable => "promptable",
            DetectorType::PromptFree => "prompt_free",
            #[allow(unreachable_patterns)]
            _ => "prompt_free",
        };
        let type_string = match detector_type {
            #[cfg(feature = "promptable")]
            DetectorType::Promptable => "promptable",
            DetectorType::PromptFree => "pf",
            #[allow(unreachable_patterns)]
            _ => "pf",
        };
        let scale_string = match scale {
            ModelScale::Nano => "n",
            ModelScale::Small => "s",
            ModelScale::Medium => "m",
            ModelScale::Large => "l",
            ModelScale::XLarge => "x",
        };
        let mask_string = if include_mask { "seg" } else { "det" };
        format!("{folder}/yoloe-26{scale_string}-{mask_string}-{type_string}.onnx")
    }

    #[must_use]
    pub fn default_prompt_free() -> Self {
        Self {
            id: Self::DEFAULT_REPO_ID.to_owned(),
            file: Self::get_model_file_path(DetectorType::PromptFree, ModelScale::Large, true),
        }
    }

    #[must_use]
    pub fn default_prompt_free_data() -> Self {
        Self {
            id: Self::DEFAULT_REPO_ID.to_owned(),
            file: format!(
                "{}.data",
                Self::get_model_file_path(DetectorType::PromptFree, ModelScale::Large, true)
            ),
        }
    }

    #[must_use]
    pub fn default_vocabulary() -> Self {
        Self {
            id: Self::DEFAULT_REPO_ID.to_owned(),
            file: "prompt_free/vocabulary_4585.json".to_owned(),
        }
    }

    #[cfg(feature = "promptable")]
    #[must_use]
    pub fn default_promptable() -> Self {
        Self {
            id: Self::DEFAULT_REPO_ID.to_owned(),
            file: Self::get_model_file_path(DetectorType::Promptable, ModelScale::Large, true),
        }
    }

    #[cfg(feature = "promptable")]
    #[must_use]
    pub fn default_promptable_data() -> Self {
        Self {
            id: Self::DEFAULT_REPO_ID.to_owned(),
            file: format!(
                "{}.data",
                Self::get_model_file_path(DetectorType::Promptable, ModelScale::Large, true)
            ),
        }
    }

    #[cfg(feature = "promptable")]
    #[must_use]
    pub fn default_clip_embedder() -> String {
        Self::DEFAULT_CLIP_REPO.to_owned()
    }
}

/// Downloads a file from `HuggingFace` Hub using the provided configuration.
#[cfg(feature = "hf-hub")]
pub async fn get_hf_model(
    model: HfModel,
    cache_dir: Option<&Path>,
) -> Result<PathBuf, ObjectDetectorError> {
    let client = match cache_dir {
        Some(dir) => HFClient::builder().cache_dir(dir).build()?,
        None => HFClient::new()?,
    };

    let (owner, name) = split_id(&model.id);
    let repo = client.model(owner, name);
    tracing::info!("Fetching {}...", &model.file);
    let downloaded_file = repo.download_file().filename(&model.file).send().await?;
    Ok(downloaded_file)
}
