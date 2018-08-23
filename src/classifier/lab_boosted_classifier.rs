// This file is part of the open-source port of SeetaFace engine, which originally includes three modules:
//      SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
//
// This file is part of the SeetaFace Detection module, containing codes implementing the face detection method described in the following paper:
//
//      Funnel-structured cascade for multi-view face detection with alignment awareness,
//      Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
//      In Neurocomputing (under review)
//
// Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
// Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
//
// As an open-source face recognition engine: you can redistribute SeetaFace source codes
// and/or modify it under the terms of the BSD 2-Clause License.
//
// You should have received a copy of the BSD 2-Clause License along with the software.
// If not, see < https://opensource.org/licenses/BSD-2-Clause>.

use std::cell::RefCell;
use std::rc::Rc;

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
        self.base_classifiers
            .push(BaseClassifier { weights, thresh })
    }
}

const K_FEAT_GROUP_SIZE: usize = 10;
const K_STDDEV_THRESH: f64 = 10.0;

impl Classifier for LabBoostedClassifier {
    fn classify(&mut self, _: Option<&mut Vec<f32>>) -> Score {
        let mut positive = true;
        let mut score = 0.0;

        let mut i = 0;
        while positive && i < self.base_classifiers.len() {
            for _ in 0..K_FEAT_GROUP_SIZE {
                let (offset_x, offset_y) = self.features[i];
                let feature_val = self
                    .feature_map
                    .borrow()
                    .get_feature_val(offset_x, offset_y);
                score += self.base_classifiers[i].weights[feature_val as usize];
                i += 1;
            }
            positive = score >= self.base_classifiers[i - 1].thresh;
        }
        positive = positive && ((*self.feature_map).borrow().get_std_dev() > K_STDDEV_THRESH);

        Score { positive, score }
    }

    fn compute(&mut self, image: &ImageData) {
        self.feature_map
            .borrow_mut()
            .compute(image.data(), image.width(), image.height());
    }

    fn set_roi(&mut self, roi: Rectangle) {
        self.feature_map.borrow_mut().set_roi(roi);
    }
}
