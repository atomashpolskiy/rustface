use std::rc::Rc;
use std::cell::RefCell;

use super::{Classifier, Score};
use common::{ImageData, Rectangle};
use feat::{FeatureMap, LabBoostedFeatureMap};

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
    fn classify(&mut self, _: Option<&mut Vec<f32>>) -> Score {
        const K_FEAT_GROUP_SIZE: usize = 10;
        const K_STDDEV_THRESH: f64 = 10f64;

        let mut positive = true;
        let mut score = 0f32;

        let mut i = 0;
        while positive && i < self.base_classifiers.len() {
            let (offset_x, offset_y) = self.features[i];
            for _ in 0..K_FEAT_GROUP_SIZE {
                let feature_val = (*self.feature_map).borrow().get_feature_val(offset_x, offset_y);
                score += self.base_classifiers[i].weights[feature_val as usize];
                i += 1;
            }
            if score < self.base_classifiers[i - 1].thresh {
                positive = false;
            }
        }
        positive = positive && ((*self.feature_map).borrow().get_std_dev() > K_STDDEV_THRESH);

        Score { positive, score }
    }

    fn compute(&mut self, image: &ImageData) {
        (*self.feature_map).borrow_mut().compute(image.data(), image.width(), image.height());
    }

    fn set_roi(&mut self, roi: Rectangle) {
        (*self.feature_map).borrow_mut().set_roi(roi);
    }
}