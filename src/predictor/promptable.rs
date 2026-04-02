use crate::predictor::nms::non_maximum_suppression;
use crate::predictor::processing::{
    preprocess_image, reconstruct_mask, ObjectBBox, ObjectDetection,
};
use crate::ObjectDetectorError;
use bon::bon;
use image::DynamicImage;
use ndarray::{s, Axis, Ix2};
use open_clip_inference::TextEmbedder;
use ort::ep::ExecutionProviderDispatch;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use rayon::prelude::*;
use std::path::Path;

#[derive(Debug)]
pub struct PromptableDetector {
    pub session: Session,
    pub text_embedder: TextEmbedder,
    image_size: u32,
    stride: u32,
}

#[bon]
impl PromptableDetector {
    #[builder]
    pub fn new(
        #[builder(start_fn)] model_path: impl AsRef<Path>,
        #[builder(start_fn)] text_embedder: TextEmbedder,
        #[builder(default = &[])] with_execution_providers: &[ExecutionProviderDispatch],
    ) -> Result<Self, ObjectDetectorError> {
        let session = Session::builder()?
            .with_execution_providers(with_execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            text_embedder,
            image_size: 640,
            stride: 32,
        })
    }

    #[builder]
    pub fn predict(
        &mut self,
        #[builder(start_fn)] img: &DynamicImage,
        #[builder(start_fn)] labels: &[&str],
        #[builder(default = 0.15)] confidence_threshold: f32,
        #[builder(default = 0.7)] intersection_over_union: f32,
    ) -> Result<Vec<ObjectDetection>, ObjectDetectorError> {
        // 1. Generate Text Embeddings
        let text_embs = self.text_embedder.embed_texts(labels)
            .map_err(|e| ObjectDetectorError::Ort(format!("CLIP error: {e}")))?;
        let text_tensor = text_embs.insert_axis(Axis(0)); // [1, N, 512]

        // 2. Preprocess Image
        let (img_tensor, meta) = preprocess_image(img, self.image_size, self.stride);

        // 3. Inference
        let outputs = self.session.run(ort::inputs![
            "images" => Value::from_array(img_tensor)?,
            "text_embeddings" => Value::from_array(text_tensor)?
        ])?;

        let raw_output = outputs["output0"].try_extract_array::<f32>()?;
        let protos = outputs["protos"].try_extract_array::<f32>()?;

        // Transpose output: [1, features, 8400] -> [8400, features]
        let preds_2d = raw_output
            .slice(s![0, .., ..])
            .into_dimensionality::<Ix2>()?
            .reversed_axes();

        let num_classes = labels.len();
        let mut candidate_boxes = Vec::new();
        let mut candidate_scores = Vec::new();
        let mut candidate_data = Vec::new();

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
                candidate_boxes.push(ObjectBBox {
                    x1: row[0] - row[2] / 2.0,
                    y1: row[1] - row[3] / 2.0,
                    x2: row[0] + row[2] / 2.0,
                    y2: row[1] + row[3] / 2.0,
                });
                candidate_scores.push(max_score);
                candidate_data.push((
                    max_cls_id,
                    row.slice(s![4 + num_classes..4 + num_classes + 32]).to_owned(),
                ));
            }
        }

        // 4. NMS
        let kept = non_maximum_suppression(&candidate_boxes, &candidate_scores, intersection_over_union);
        let protos_view = protos.slice(s![0, .., .., ..]);

        // 5. Post-process coordinates and masks
        Ok(kept
            .into_par_iter()
            .map(|idx| {
                let (class_id, weights) = &candidate_data[idx];
                let raw_box = candidate_boxes[idx];

                let final_bbox = ObjectBBox {
                    x1: ((raw_box.x1 - meta.pad.0) / meta.ratio).clamp(0.0, meta.orig_shape.0 as f32),
                    y1: ((raw_box.y1 - meta.pad.1) / meta.ratio).clamp(0.0, meta.orig_shape.1 as f32),
                    x2: ((raw_box.x2 - meta.pad.0) / meta.ratio).clamp(0.0, meta.orig_shape.0 as f32),
                    y2: ((raw_box.y2 - meta.pad.1) / meta.ratio).clamp(0.0, meta.orig_shape.1 as f32),
                };

                ObjectDetection {
                    bbox: final_bbox,
                    score: candidate_scores[idx],
                    class_id: *class_id,
                    tag: labels[*class_id].to_string(),
                    mask: Some(reconstruct_mask(&protos_view, weights, &meta, &final_bbox)),
                }
            })
            .collect())
    }
}