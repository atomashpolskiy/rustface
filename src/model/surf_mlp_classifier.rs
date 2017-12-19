use feat::FeatureMap;
use std::rc::Rc;
use super::classifier::{Classifier, Score};

pub struct SurfMlpClassifier {
    feature_map: Rc<FeatureMap>,
    feature_ids: Vec<i32>,
    thresh: f32,
    layers: Vec<Layer>,
}

struct Layer {

}

impl SurfMlpClassifier {
    pub fn new(feature_map: Rc<FeatureMap>) -> Self {
        SurfMlpClassifier {
            feature_map,
            feature_ids: vec![],
            thresh: 0f32,
            layers: vec![],
        }
    }

    pub fn add_feature_id(&mut self, feature_id: i32) {
        self.feature_ids.push(feature_id);
    }

    pub fn set_threshold(&mut self, thresh: f32) {
        self.thresh = thresh;
    }

    pub fn add_layer(&mut self, input_dim: i32, output_dim: i32, weights: Vec<f32>, biases: Vec<f32>) {

    }

    pub fn add_output_layer(&mut self, input_dim: i32, output_dim: i32, weights: Vec<f32>, biases: Vec<f32>) {

    }
}

impl Classifier for SurfMlpClassifier {
    fn classify(&self, features: FeatureMap) -> Option<Score> {
        unimplemented!()
    }
}