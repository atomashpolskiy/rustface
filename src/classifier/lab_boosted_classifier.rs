use std::rc::Rc;
use std::cell::RefCell;

use super::{Classifier, Score};
use feat::LabBoostedFeatureMap;

pub struct LabBoostedClassifier {
    feature_map: Rc<RefCell<LabBoostedFeatureMap>>,
    features: Vec<(i32, i32)>,
    base_classifiers: Vec<BaseClassifier>,
}

struct BaseClassifier {
    weights: Vec<f32>,
    thresh: f32,
}

impl LabBoostedClassifier {
    pub fn new(feature_map: Rc<RefCell<LabBoostedFeatureMap>>) -> Self {
        LabBoostedClassifier {
            feature_map,
            features: vec![],
            base_classifiers: vec![],
        }
    }

    pub fn add_feature(&mut self, x: i32, y: i32) {
        self.features.push((x, y));
    }

    pub fn add_base_classifier(&mut self, weights: Vec<f32>, thresh: f32) {
        self.base_classifiers.push(
            BaseClassifier { weights, thresh }
        )
    }
}

impl Classifier for LabBoostedClassifier {
    fn classify(&mut self, output: &mut Vec<f32>) -> Score {
        unimplemented!()
    }
}