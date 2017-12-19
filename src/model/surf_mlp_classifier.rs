use feat::FeatureMap;
use std::rc::Rc;
use super::classifier::{Classifier, Score};
use math;

pub struct SurfMlpClassifier {
    feature_map: Rc<FeatureMap>,
    feature_ids: Vec<i32>,
    thresh: f32,
    layers: Vec<Layer>,
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

type ActFunc = Fn(f32) -> f32;

struct Layer {
    input_dim: i32,
    weights: Vec<f32>,
    biases: Vec<f32>,
    act_func: Box<ActFunc>,
}

impl Layer {
    fn compute(&self, input: &Vec<f32>, output: &mut Vec<f32>) {
        let input_dim = self.input_dim as usize;
        let output_dim = self.biases.len();
        for i in 0..output_dim as usize {
            let x;
            unsafe {
                x = math::vector_inner_product(
                    input.as_ptr(),
                    self.weights.as_ptr().offset((i * input_dim) as isize),
                    input_dim);
            }
            output[i] = (self.act_func)(x);
        }
    }
}

impl Classifier for SurfMlpClassifier {
    fn classify(&self, features: FeatureMap) -> Option<Score> {
        unimplemented!()
    }
}