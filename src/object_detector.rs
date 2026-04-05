use crate::ObjectDetectorError;
use crate::model_manager::{HfModel, get_hf_model};
use crate::predictor::{PromptFreeDetector, PromptableDetector};
use crate::structs::{DetectedObject, DetectorType, ModelScale};
use bon::bon;
use image::DynamicImage;
use ort::ep::ExecutionProviderDispatch;

pub struct ObjectDetector {
    inner: ObjectDetectorInner,
}

enum ObjectDetectorInner {
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
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
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

        let model_path_local = get_hf_model(model).await?;
        get_hf_model(data_model).await?;

        let inner = match detector_type {
            DetectorType::Promptable => {
                let text_embedder =
                    open_clip_inference::TextEmbedder::from_hf(&HfModel::default_clip_embedder())
                        .with_execution_providers(with_execution_providers)
                        .build()
                        .await
                        .map_err(|e| ObjectDetectorError::Ort(format!("CLIP error: {e}")))?;

                let detector = PromptableDetector::builder(model_path_local, text_embedder)
                    .with_execution_providers(with_execution_providers)
                    .build()?;
                ObjectDetectorInner::Promptable(Box::new(detector))
            }
            DetectorType::PromptFree => {
                let vocab_model = HfModel::default_vocabulary();
                let vocab_path = get_hf_model(vocab_model).await?;

                let detector = PromptFreeDetector::builder(model_path_local, vocab_path)
                    .with_execution_providers(with_execution_providers)
                    .build()?;
                ObjectDetectorInner::PromptFree(detector)
            }
        };

        Ok(Self { inner })
    }

    #[builder]
    pub fn predict(
        &mut self,
        #[builder(start_fn)] img: &DynamicImage,
        #[builder(default = &[])] labels: &[&str],
        #[builder(default = 0.4)] confidence_threshold: f32,
        #[builder(default = 0.7)] intersection_over_union: f32,
    ) -> Result<Vec<DetectedObject>, ObjectDetectorError> {
        match &mut self.inner {
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
        }
    }
}
