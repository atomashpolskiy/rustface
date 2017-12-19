use feat::FeatureMap;
use std::rc::Rc;
use super::classifier::{Classifier, Score};

pub struct SurfMlpClassifier {
    feature_map: Rc<FeatureMap>,
    feature_ids: Vec<i32>,
    thresh: f32,
    layers: Vec<Layer>,
}

type ActFunc = Fn(f32) -> f32;

struct Layer {
    input_dim: i32,
    weights: Vec<f32>,
    biases: Vec<f32>,
    act_func: Box<ActFunc>,
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
        self.layers.push(
            Layer {
                input_dim, weights, biases,
                act_func: Box::new(Self::relu)
            }
        )
    }

    pub fn add_output_layer(&mut self, input_dim: i32, output_dim: i32, weights: Vec<f32>, biases: Vec<f32>) {
        self.layers.push(
            Layer {
                input_dim, weights, biases,
                act_func: Box::new(Self::sigmoid)
            }
        )
    }

    fn relu(x: f32) -> f32 {
        if x > 0f32 {
            x
        } else {
            0f32
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1f32 / (1f32 + (-x).exp())
    }
}

impl Classifier for SurfMlpClassifier {
    fn classify(&self, features: FeatureMap) -> Option<Score> {
        unimplemented!()
    }
}