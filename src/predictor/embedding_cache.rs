use crate::ObjectDetectorError;
use ndarray::{Array1, Array2, Axis, stack};
use open_clip_inference::TextEmbedder;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

#[derive(Default, Debug)]
pub struct EmbeddingCache {
    /// Inner map: Model Directory -> (Label -> Embedding Vector)
    cache: RwLock<HashMap<PathBuf, HashMap<String, Array1<f32>>>>,
}

impl EmbeddingCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_embed(
        &self,
        labels: &[&str],
        embedder: &TextEmbedder,
    ) -> Result<Array2<f32>, ObjectDetectorError> {
        let model_key = embedder.model_dir.clone();

        // Quick check for existing embeddings (Read Lock)
        {
            let read_guard = self
                .cache
                .read()
                .map_err(|e| ObjectDetectorError::Ort(e.to_string()))?;
            if let Some(model_cache) = read_guard.get(&model_key) {
                if labels.iter().all(|l| model_cache.contains_key(*l)) {
                    return self.assemble_from_cache(labels, model_cache);
                }
            }
        }

        // Determine what needs to be embedded
        let mut missing_labels = Vec::new();
        {
            let read_guard = self
                .cache
                .read()
                .map_err(|e| ObjectDetectorError::Ort(e.to_string()))?;
            let model_cache = read_guard.get(&model_key);
            for &label in labels {
                if model_cache.map_or(true, |m| !m.contains_key(label)) {
                    missing_labels.push(label);
                }
            }
        }

        // Batch embed missing labels
        if !missing_labels.is_empty() {
            let new_embeddings = embedder
                .embed_texts(&missing_labels)
                .map_err(|e| ObjectDetectorError::Ort(format!("CLIP error: {e}")))?;

            // Update Cache
            let mut write_guard = self
                .cache
                .write()
                .map_err(|e| ObjectDetectorError::Ort(e.to_string()))?;
            let model_cache = write_guard.entry(model_key).or_default();

            for (i, label) in missing_labels.into_iter().enumerate() {
                let emb = new_embeddings.index_axis(Axis(0), i).to_owned();
                model_cache.insert(label.to_string(), emb);
            }
        }

        let read_guard = self
            .cache
            .read()
            .map_err(|e| ObjectDetectorError::Ort(e.to_string()))?;
        let model_cache = read_guard.get(&embedder.model_dir).ok_or_else(|| {
            ObjectDetectorError::Ort("Cache inconsistency after update".to_string())
        })?;

        self.assemble_from_cache(labels, model_cache)
    }

    fn assemble_from_cache(
        &self,
        labels: &[&str],
        model_cache: &HashMap<String, Array1<f32>>,
    ) -> Result<Array2<f32>, ObjectDetectorError> {
        let views: Vec<_> = labels
            .iter()
            .map(|&l| {
                model_cache
                    .get(l)
                    .map(|arr| arr.view())
                    .ok_or_else(|| ObjectDetectorError::Ort(format!("Label not found: {l}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        stack(Axis(0), &views).map_err(ObjectDetectorError::NdArray)
    }
}
