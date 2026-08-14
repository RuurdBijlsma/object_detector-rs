use crate::ObjectDetectorError;
#[cfg(feature = "hf-hub")]
use crate::model_manager::{HfModel, get_hf_model};
use crate::predictor::PromptFreeDetector;
#[cfg(feature = "promptable")]
use crate::predictor::PromptableDetector;
use crate::structs::{DetectedObject, DetectorType, ModelScale};
use bon::bon;
use image::DynamicImage;
use ort::ep::ExecutionProviderDispatch;
use ort::session::builder::GraphOptimizationLevel;
use std::path::Path;

pub struct ObjectDetector {
    inner: ObjectDetectorInner,
}

enum ObjectDetectorInner {
    #[cfg(feature = "promptable")]
    Promptable(Box<PromptableDetector>),
    PromptFree(PromptFreeDetector),
}

#[bon]
impl ObjectDetector {
    /// Initialize predictor using models hosted on Hugging Face.
    #[cfg(feature = "hf-hub")]
    #[builder(finish_fn = build)]
    pub async fn from_hf(
        #[builder(start_fn)] detector_type: DetectorType,
        #[builder(default = ModelScale::Large)] scale: ModelScale,
        #[builder(default = true)] include_mask: bool,
        cache_dir: Option<&Path>,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
        with_intra_threads: Option<usize>,
        with_inter_threads: Option<usize>,
        with_memory_pattern: Option<bool>,
        with_optimization_level: Option<GraphOptimizationLevel>,
    ) -> Result<Self, ObjectDetectorError> {
        let model_path = HfModel::get_model_file_path(detector_type, scale, include_mask);
        let model = HfModel {
            id: HfModel::DEFAULT_REPO_ID.to_owned(),
            file: model_path.clone(),
        };
        let data_model = HfModel {
            id: HfModel::DEFAULT_REPO_ID.to_owned(),
            file: format!("{model_path}.data"),
        };

        let model_path_local = get_hf_model(model, cache_dir).await?;
        get_hf_model(data_model, cache_dir).await?;

        let inner = match detector_type {
            #[cfg(feature = "promptable")]
            DetectorType::Promptable => {
                let text_embedder =
                    open_clip_inference::TextEmbedder::from_hf(&HfModel::default_clip_embedder())
                        .maybe_cache_dir(cache_dir)
                        .with_execution_providers(with_execution_providers)
                        .maybe_with_intra_threads(with_intra_threads)
                        .maybe_with_inter_threads(with_inter_threads)
                        .maybe_with_memory_pattern(with_memory_pattern)
                        .maybe_with_optimization_level(with_optimization_level)
                        .build()
                        .await
                        .map_err(|e| ObjectDetectorError::Ort(format!("CLIP error: {e}")))?;

                let detector = PromptableDetector::builder(model_path_local, text_embedder)
                    .with_execution_providers(with_execution_providers)
                    .maybe_with_intra_threads(with_intra_threads)
                    .maybe_with_inter_threads(with_inter_threads)
                    .maybe_with_memory_pattern(with_memory_pattern)
                    .maybe_with_optimization_level(with_optimization_level)
                    .build()?;
                ObjectDetectorInner::Promptable(Box::new(detector))
            }
            DetectorType::PromptFree => {
                let vocab_model = HfModel::default_vocabulary();
                let vocab_path = get_hf_model(vocab_model, cache_dir).await?;

                let detector = PromptFreeDetector::builder(model_path_local, vocab_path)
                    .with_execution_providers(with_execution_providers)
                    .maybe_with_intra_threads(with_intra_threads)
                    .maybe_with_inter_threads(with_inter_threads)
                    .maybe_with_memory_pattern(with_memory_pattern)
                    .maybe_with_optimization_level(with_optimization_level)
                    .build()?;
                ObjectDetectorInner::PromptFree(detector)
            }
            #[allow(unreachable_patterns)]
            _ => {
                return Err(ObjectDetectorError::InvalidModel(
                    "Promptable detector is disabled".into(),
                ));
            }
        };

        Ok(Self { inner })
    }

    #[builder]
    pub fn predict(
        &self,
        #[builder(start_fn)] img: &DynamicImage,
        #[builder(default = &[])] labels: &[&str],
        #[builder(default = 0.3)] confidence_threshold: f32,
        #[builder(default = 0.7)] intersection_over_union: f32,
    ) -> Result<Vec<DetectedObject>, ObjectDetectorError> {
        match &self.inner {
            #[cfg(feature = "promptable")]
            ObjectDetectorInner::Promptable(detector) => {
                if labels.is_empty() {
                    return Err(ObjectDetectorError::InvalidModel(
                        "Labels are required for Promptable detector".into(),
                    ));
                }
                detector
                    .predict(img, labels)
                    .confidence_threshold(confidence_threshold)
                    .intersection_over_union(intersection_over_union)
                    .call()
            }
            ObjectDetectorInner::PromptFree(detector) => {
                if !labels.is_empty() {
                    return Err(ObjectDetectorError::InvalidModel(
                        "Labels are not supported for PromptFree detector".into(),
                    ));
                }
                detector
                    .predict(img)
                    .confidence_threshold(confidence_threshold)
                    .intersection_over_union(intersection_over_union)
                    .call()
            }
            #[allow(unreachable_patterns)]
            _ => Err(ObjectDetectorError::InvalidModel(
                "Promptable detector is disabled".into(),
            )),
        }
    }
}
