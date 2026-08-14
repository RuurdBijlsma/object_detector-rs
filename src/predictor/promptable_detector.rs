// Mutex lock drop can't be done earlier
#![allow(clippy::significant_drop_tightening)]
use crate::ObjectDetectorError;
#[cfg(feature = "hf-hub")]
use crate::model_manager::{HfModel, get_hf_model};
use crate::onnx::OnnxSession;
use crate::predictor::EmbeddingCache;
use crate::predictor::nms::non_maximum_suppression;
use crate::predictor::processing::{Candidate, YoloEngine, finalize_detections, preprocess_image};
use crate::structs::{DetectedObject, ObjectBBox};
use bon::bon;
use image::DynamicImage;
use ndarray::{Array1, Axis, Ix2, s};
use open_clip_inference::TextEmbedder;
use ort::ep::ExecutionProviderDispatch;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

#[derive(Debug)]
pub struct PromptableDetector {
    engine: YoloEngine,
    pub text_embedder: TextEmbedder,
    cache: EmbeddingCache,
}

#[bon]
impl PromptableDetector {
    /// Initialize predictor using models hosted on Hugging Face.
    #[cfg(feature = "hf-hub")]
    #[builder(finish_fn = build)]
    pub async fn from_hf(
        #[builder(default = HfModel::default_promptable())] model: HfModel,
        #[builder(default = HfModel::default_promptable_data())] data_model: HfModel,
        #[builder(default = HfModel::default_clip_embedder())] clip_hf_repo: String,
        cache_dir: Option<&Path>,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
        with_intra_threads: Option<usize>,
        with_inter_threads: Option<usize>,
        with_memory_pattern: Option<bool>,
        with_optimization_level: Option<GraphOptimizationLevel>,
    ) -> Result<Self, ObjectDetectorError> {
        let model_path = get_hf_model(model, cache_dir).await?;
        get_hf_model(data_model, cache_dir).await?;
        let text_embedder = TextEmbedder::from_hf(&clip_hf_repo)
            .maybe_cache_dir(cache_dir)
            .with_execution_providers(with_execution_providers)
            .maybe_with_intra_threads(with_intra_threads)
            .maybe_with_inter_threads(with_inter_threads)
            .maybe_with_memory_pattern(with_memory_pattern)
            .maybe_with_optimization_level(with_optimization_level)
            .build()
            .await?;
        Self::builder(model_path, text_embedder)
            .with_execution_providers(with_execution_providers)
            .maybe_with_intra_threads(with_intra_threads)
            .maybe_with_inter_threads(with_inter_threads)
            .maybe_with_memory_pattern(with_memory_pattern)
            .maybe_with_optimization_level(with_optimization_level)
            .build()
    }

    #[builder]
    pub fn new(
        #[builder(start_fn)] model_path: impl AsRef<Path>,
        #[builder(start_fn)] text_embedder: TextEmbedder,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
        with_intra_threads: Option<usize>,
        with_inter_threads: Option<usize>,
        with_memory_pattern: Option<bool>,
        with_optimization_level: Option<GraphOptimizationLevel>,
    ) -> Result<Self, ObjectDetectorError> {
        let onnx = OnnxSession::new(
            model_path,
            with_execution_providers,
            with_optimization_level,
            with_intra_threads,
            with_inter_threads,
            with_memory_pattern,
        )?;

        Ok(Self {
            engine: YoloEngine {
                session: Mutex::new(onnx.session),
                image_size: 640,
                stride: 32,
            },
            text_embedder,
            cache: EmbeddingCache::new(),
        })
    }

    #[builder]
    pub fn predict(
        &self,
        #[builder(start_fn)] img: &DynamicImage,
        #[builder(start_fn)] labels: &[&str],
        #[builder(default = 0.2)] confidence_threshold: f32,
        #[builder(default = 0.7)] intersection_over_union: f32,
    ) -> Result<Vec<DetectedObject>, ObjectDetectorError> {
        let text_embs = self.cache.get_or_embed(labels, &self.text_embedder)?;
        let text_tensor = text_embs.insert_axis(Axis(0)); // [1, N, 512]

        let (img_tensor, meta) = preprocess_image(img, self.engine.image_size, self.engine.stride);

        // Inference
        let mut session = self.engine.session.lock()?;
        let outputs = session.run(ort::inputs![
            "images" => Value::from_array(img_tensor)?,
            "text_embeddings" => Value::from_array(text_tensor)?
        ])?;

        let raw_output = outputs["output0"].try_extract_array::<f32>()?;
        let protos = outputs
            .get("protos")
            .map(|p| p.try_extract_array::<f32>())
            .transpose()?;

        // Transpose output: [1, features, 8400] -> [8400, features]
        let preds_2d = raw_output
            .slice(s![0, .., ..])
            .into_dimensionality::<Ix2>()?
            .reversed_axes();

        let num_classes = labels.len();

        // Check if model has enough columns for mask weights (4 box + num_classes + 32 weights)
        let has_masks = protos.is_some() && preds_2d.shape()[1] >= 4 + num_classes + 32;

        let mut candidates = Vec::new();

        // Extract candidates
        for i in 0..preds_2d.shape()[0] {
            let row = preds_2d.row(i);
            let scores = row.slice(s![4..4 + num_classes]);

            let mut max_score = 0.0f32;
            let mut max_cls_id = 0;
            for (idx, &s) in scores.iter().enumerate() {
                if s > max_score {
                    max_score = s;
                    max_cls_id = idx;
                }
            }

            if max_score > confidence_threshold {
                let mask_weights = if has_masks {
                    row.slice(s![4 + num_classes..4 + num_classes + 32])
                        .to_owned()
                } else {
                    Array1::default(0)
                };

                candidates.push(Candidate {
                    bbox: ObjectBBox {
                        x1: row[0] - row[2] / 2.0,
                        y1: row[1] - row[3] / 2.0,
                        x2: row[0] + row[2] / 2.0,
                        y2: row[1] + row[3] / 2.0,
                    },
                    score: max_score,
                    class_id: max_cls_id,
                    mask_weights,
                });
            }
        }

        // NMS
        let bboxes: Vec<_> = candidates.iter().map(|c| c.bbox).collect();
        let scores: Vec<_> = candidates.iter().map(|c| c.score).collect();
        let kept_indices = non_maximum_suppression(&bboxes, &scores, intersection_over_union);

        let kept_candidates: Vec<Candidate> = kept_indices
            .into_iter()
            .map(|idx| candidates[idx].clone())
            .collect();

        // Prepare protos view if it exists
        let protos_view = protos.as_ref().map(|p| p.slice(s![0, .., .., ..]));

        // Convert slice labels to String for the shared finalizer
        let label_strings: Vec<String> = labels.iter().map(ToString::to_string).collect();

        Ok(finalize_detections(
            kept_candidates,
            protos_view.as_ref(),
            &meta,
            &label_strings,
        ))
    }
}
